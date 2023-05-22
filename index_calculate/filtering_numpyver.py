import math
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import pywt

TIFF_DIR_PATH = "D:\\study\\yanyi\\yaogan\\week2\\MOD09A1\\EVI2"
INVALID_VALUE = 20
LAMBDA = 5000
WINDOW_LENGTH = 51
OUTPUT_PATH = ""


def main():
    test_row = 0
    test_col = 0
    tiff_files = get_tiff_files()
    doy = get_doy(tiff_files)
    data = get_tiff_data_arrays(tiff_files, start_row=100, start_col=100, row_num=50, col_num=50)
    plt.plot(doy, data[0:, test_row, test_col], 'o', label='original data')
    interpolated_data = interpolate_arrays(data, doy)
    plt.plot(range(1, len(interpolated_data) + 1), interpolated_data[0:, test_row, test_col], label='interpolated data')
    filtered_data = smooth_filter_arrays(interpolated_data, "W")
    plt.plot(range(1, len(filtered_data) + 1), filtered_data[0:, test_row, test_col], label='smoothed data')
    peaks, peak_heights = signal.find_peaks(filtered_data[0:, test_row, test_col], height=0.3, distance=60, prominence=0.1)
    plt.plot(peaks, filtered_data[0:, test_row, test_col][peaks], "x", label='heading date')
    ci = calculate_cropping_intensity(filtered_data, height=0.3, distance=60, prominence=0.1)
    # save_as_tiff_2d(ci, "multiple crop index")
    plt.legend()
    plt.show()
    save_as_tiff(filtered_data, "W")
    return


def get_tiff_files():
    """
    read tiff file from a specified directory, and convert them into objects.

    :param:
      none.

    :return tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object
    """

    tiff_file_name_list = []
    tiff_file_object_dict = {}
    for file_name in os.listdir(TIFF_DIR_PATH):
        if file_name[-5:] == ".tiff":
            tiff_file_name_list.append(file_name)
    total_progress = len(tiff_file_name_list)
    tqdm.write("found " + str(total_progress) + " tiff files.")
    tqdm.write("start converting tiff files to objects.\nsource directory path: " + TIFF_DIR_PATH)
    progress_bar = tqdm(range(total_progress))
    for file_name in tiff_file_name_list:
        tiff_file_object_dict[file_name[13:16]] = Image.open(TIFF_DIR_PATH + "\\" + file_name)
        progress_bar.update(1)
    progress_bar.close()
    tqdm.write("done.\n_______________________________________________________")
    return tiff_file_object_dict


def get_tiff_data_arrays(tiff_file_object_dict, start_row=0, start_col=0, row_num=2400, col_num=2400):
    """
    convert the data in the tiff file(all or part of the size)
    into a numpy 3d-array that can be processed.
    shape of 3d-array: (tiff file number, row number, column number).

    :param tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object

    :param start_row:
      start row index

    :param start_col:
      start column index

    :param row_num:
      the number of rows to be processed

    :param col_num:
      the number of column to be processed

    :return tiff_data_arrays:
      a numpy 3d-array
      array element type: float 32
      shape: (tiff file number, row number, column number)
    """

    tqdm.write("start extracting data from tiff files.")
    end_row = start_row + row_num
    end_col = start_col + col_num
    tqdm.write("row: " + str(start_row) + " to " + str(end_row - 1))
    tqdm.write("column: " + str(start_col) + " to " + str(end_col - 1))
    tiff_data_arrays = [np.array(array)[start_row:end_row, start_col:end_col]
                        for array in list(tiff_file_object_dict.values())]
    tiff_data_arrays = np.stack(tiff_data_arrays)
    tqdm.write("done.\n_______________________________________________________")
    return tiff_data_arrays


def interpolate_arrays(data_arrays, doy):
    """
    linearly interpolate the data.

    :param data_arrays:
      type: numpy 3d-array
      array element type: float 32
      shape: (tiff file number, row number, column number)

    :param doy:
      type: a list of int
      tiff's DOY

    :return interpolated_arrays:
      interpolated data.
      type: a list of numpy 3d-arrays
      array element type: float 32
      shape: (interpolated number of days, row number, column number)
    """

    x_interpolated = np.arange(doy[0], doy[-1] + 1, 1)
    row_num = data_arrays.shape[1]
    col_num = data_arrays.shape[2]
    interpolated_arrays = np.empty((len(x_interpolated), row_num, col_num))
    total_progress = row_num * col_num
    progress_bar = tqdm(range(total_progress))
    tqdm.write("start interpolating data.")

    for row in range(row_num):
        for col in range(col_num):
            y = data_arrays[:, row, col]
            y = y.tolist()
            x = doy.tolist()

            # remove nan data in each row
            index_to_delete = []
            for index, number in enumerate(y):
                if math.isnan(number):
                    index_to_delete.append(index)
            for counter, index in enumerate(index_to_delete):
                index = index - counter
                x.pop(index)
                y.pop(index)

            # if there are more than 10 nan data in a row,
            # all the data in this row will be set to invalid values
            if len(doy) - len(x) > 10:
                # tqdm.write("filled with invalid value: row: " + str(row) + " col: " + str(col))
                interpolated_arrays[:, row, col] = np.array([INVALID_VALUE] * len(x_interpolated))
                progress_bar.update(1)
                continue

            # handle the special case where the first or last value is nan
            if 0 in index_to_delete:
                y.insert(0, y[0] + (doy[0] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
                x.insert(0, doy[0])
            if len(doy) - 1 in index_to_delete:
                y.append(y[-1] + (doy[-1] - x[-1]) * (y[-2] - y[-1]) / (x[-2] - x[-1]))
                x.append(doy[-1])

            f = interpolate.interp1d(np.array(x), np.array(y))
            interpolated_arrays[:, row, col] = f(x_interpolated)
            progress_bar.update(1)
    progress_bar.close()
    tqdm.write("done.\n_______________________________________________________")
    return interpolated_arrays


def get_doy(tiff_file_object_dict):
    """
    get DOY sequence of tiff files.

    :param tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object

    :return:
      DOY sequence of tiff files
      type: a list of int
    """
    return np.array(list(map(int, list(tiff_file_object_dict.keys()))))


def smooth_filter_arrays(data_arrays, filter_type):
    """
    smooth filter the data.

    :param data_arrays:
      data to be filtered.
      type: numpy 3d-array
      array element type: float 32
      shape: (interpolated number of days, row number, column number)

    :param filter_type:
      type: string
      the type of smoothing filter to apply to the data.
      options:
        W(Whittaker smoother) / SG(Savitzky-Golay filter)

    :return filtered_arrays:
      filtered data.
      type: a list of numpy 3d-arrays
      array element type: float 32
      shape: (interpolated number of days, row number, column number)

    """

    row_num = data_arrays.shape[1]
    col_num = data_arrays.shape[2]
    filtered_arrays = np.empty(data_arrays.shape)
    total_progress = row_num * col_num
    progress_bar = tqdm(range(total_progress))
    tqdm.write("start filtering data.")

    # Whittaker smoother
    # source: Whittaker Smoother by Neal B. Gallagher
    if filter_type == "W":
        for row in range(row_num):
            for col in range(col_num):
                y = data_arrays[:, row, col]
                m = len(y)
                d = np.diff(np.eye(m), 2)
                w = np.diag([int(not math.isnan(i)) for i in y])
                z = np.dot(np.linalg.inv(w + LAMBDA * np.dot(d, np.transpose(d))), (np.dot(w, y)))
                filtered_arrays[:, row, col] = z
                progress_bar.update(1)
    # Savitzky-Golay filter
    # source: A method for reconstructing NDVI time-series based on envelope detection and the Savitzky-Golay filter
    elif filter_type == "SG":
        for row in range(row_num):
            for col in range(col_num):
                y = data_arrays[:, row, col]
                r = signal.savgol_filter(y, WINDOW_LENGTH, 2)
                filtered_arrays[:, row, col] = r
                progress_bar.update(1)

    # elif filter_type == "CWT":
    #     for row in range(row_num):
    #         for col in range(col_num):
    #             y = data_arrays[:, row, col]
    #             coef, freqs = pywt.cwt(y, np.arange(10, 41), "morl")
    #             filtered_arrays[:, row, col] = pywt.waverec(coef, "morl")
    #             progress_bar.update(1)

    progress_bar.close()
    tqdm.write("done.\n_______________________________________________________")
    return filtered_arrays


def calculate_cropping_intensity(data_arrays, height=0.3, distance=60, prominence=0.1):
    """
    calculate multiple crop index, base on filtered data.

    :param data_arrays:
      type: numpy 3d-array
      array element type: float 32
      shape: (interpolated number of days, row number, column number)

    :param height:
      required height of peaks.

    :param distance:
      required minimal horizontal distance (>= 1) in samples between neighbouring peaks.

    :param prominence:
      Required prominence of peaks.

    :return multiple_crop_index_arrays:
      type: numpy 2d-array
      array element type: int
      shape: (row number, column number)
    """

    row_num = data_arrays.shape[1]
    col_num = data_arrays.shape[2]
    cropping_intensity_arrays = np.empty((row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            data = data_arrays[:, row, col]
            peaks, peak_heights = signal.find_peaks(data, height=height, distance=distance, prominence=prominence)
            cropping_intensity_arrays[row, col] = len(peaks)
    # print(cropping_intensity_arrays)
    return cropping_intensity_arrays


def save_as_tiff(data_arrays, filter_type):
    """
    save data as tiff file.

    :param data_arrays:
      type: numpy 3d-array
      array element type: float 32
      shape: (interpolated number of days, row number, column number)

    :param filter_type:
      type: string
      the type of smoothing filter to applied to the data.
      options:
        W(Whittaker smoother) / SG(Savitzky-Golay filter)

    :return:
      none.
    """

    row_num = data_arrays.shape[1]
    col_num = data_arrays.shape[2]
    total_progress = data_arrays.shape[0]
    progress_bar = tqdm(range(total_progress))
    tqdm.write("start saving data to tiff file.")

    driver = gdal.GetDriverByName('GTiff')
    output_path = ""
    if OUTPUT_PATH == "":
        output_path = TIFF_DIR_PATH + "\\" + filter_type + "filtered"
    else:
        output_path = OUTPUT_PATH
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tqdm.write("output directory path: " + output_path)

    for index, array in enumerate(data_arrays):
        output_file_name = output_path + "\\" + str(index + 1) + ".tiff"
        created_raster = driver.Create(output_file_name, col_num, row_num, 1, gdal.GDT_Float32)
        # created_raster.SetNoDataValue(INVALID_VALUE)
        created_raster.GetRasterBand(1).SetNoDataValue(INVALID_VALUE)
        created_raster.GetRasterBand(1).WriteArray(array)
        progress_bar.update(1)
    progress_bar.close()
    tqdm.write("done.\n_______________________________________________________")
    return


def save_as_tiff_2d(data_array, filter_type):
    """
    save data as tiff file.

    :param data_array:
      type: numpy 2d-array
      array element type: float 32
      shape: (row number, column number)

    :param filter_type:
      type: string
      the type of smoothing filter to applied to the data.

    :return:
      none.
    """

    row_num = data_array.shape[0]
    col_num = data_array.shape[1]
    tqdm.write("start saving data to tiff file.")

    driver = gdal.GetDriverByName('GTiff')
    output_path = ""
    if OUTPUT_PATH == "":
        output_path = TIFF_DIR_PATH + "\\" + filter_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tqdm.write("output directory path: " + output_path)

    output_file_name = output_path + "\\" + filter_type + ".tiff"
    created_raster = driver.Create(output_file_name, col_num, row_num, 1, gdal.GDT_Float32)
    # created_raster.SetNoDataValue(INVALID_VALUE)
    created_raster.GetRasterBand(1).SetNoDataValue(INVALID_VALUE)
    created_raster.GetRasterBand(1).WriteArray(data_array)
    tqdm.write("done.\n_______________________________________________________")
    return


if __name__ == '__main__':
    main()
