import math
import numpy
import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import os
from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import pywt

FILTERED_DIR_PATH_EVI2 = "D:\\BaiduNetdiskDownload\\EVI2"
FILTERED_DIR_PATH_LSWI = "D:\\BaiduNetdiskDownload\\LSWI"
OUTPUT_PATH = "D:\\study\\yanyi\\yaogan\\remote_sensing\\mapping"


def main():

    # file = get_tiff_files(FILTERED_DIR_PATH_EVI2)
    # data = get_tiff_data_arrays(file, start_row=0, start_col=0, row_num=1198, col_num=1839)
    # ci = calculate_cropping_intensity(data, distance=80)
    # save_as_tiff_2d(ci, 'ci2')
    # test_row = 1157
    # test_col = 264
    # plt.plot(range(1, len(data) + 1), data[0:, test_row, test_col], label='smoothed data')
    # peaks, peak_heights = signal.find_peaks(data[0:, test_row, test_col], height=0.3, distance=60,
    #                                         prominence=0.1)
    # plt.plot(peaks, data[0:, test_row, test_col][peaks], "x", label='heading date')
    # plt.legend()
    # plt.show()

    # save_as_tiff_2d(mapping_paddy_rice(), "padding_rice")
    save_as_tiff_2d(mapping_winter_wheat(), "winter wheat")
    return


def mapping_paddy_rice():
    """
    source: Mapping paddy rice areas based on vegetation phenology and surface moisture conditions
    Bingwen Qiu(2015)

    :param:
      none.

    :return padding_rice_map:
      a numpy 2d-array
      array element type: float 32
      shape: (row number, column number)
    """
    test_row = 0
    test_col = 0

    evi2_tiff_files = get_tiff_files(FILTERED_DIR_PATH_EVI2)
    evi2_data = get_tiff_data_arrays(evi2_tiff_files, start_row=0, start_col=0, row_num=1198, col_num=1839)
    row_num = evi2_data.shape[1]
    col_num = evi2_data.shape[2]
    plt.plot(range(1, len(evi2_data) + 1), evi2_data[:, test_row, test_col], label='smoothed EVI2 data', c="royalblue")
    heading_date_arrays = np.empty((row_num, col_num), dtype=numpy.int32)
    evi2_heading = np.empty((row_num, col_num))
    evi2_tillering = np.empty((row_num, col_num))

    for row in range(row_num):
        for col in range(col_num):
            data = evi2_data[:, row, col]
            peaks, peak_heights = signal.find_peaks(data, height=0.3, distance=60, prominence=0.1)
            if len(peaks) >= 1:
                if peaks[0] >= 40:
                    heading_date_arrays[row, col] = peaks[0]
                else:
                    heading_date_arrays[row, col] = -1
            else:
                heading_date_arrays[row, col] = -1

    for row in range(row_num):
        for col in range(col_num):
            if not heading_date_arrays[row, col] == -1:
                evi2_heading[row, col] = evi2_data[heading_date_arrays[row, col], row, col]
                evi2_tillering[row, col] = evi2_data[heading_date_arrays[row, col] - 40, row, col]

    plt.plot(heading_date_arrays[test_row, test_col] + 1, evi2_heading[test_row, test_col], 'o', c="royalblue")
    plt.axvline(heading_date_arrays[test_row, test_col] + 1, ls="--", c="black")
    plt.plot(heading_date_arrays[test_row, test_col] - 40 + 1, evi2_tillering[test_row, test_col], 'o', c="royalblue")
    plt.axvline(heading_date_arrays[test_row, test_col] - 40 + 1, ls="--", c="black")

    lswi_tiff_files = get_tiff_files(FILTERED_DIR_PATH_LSWI)
    lswi_data = get_tiff_data_arrays(lswi_tiff_files, start_row=0, start_col=0, row_num=1198, col_num=1839)
    plt.plot(range(1, len(lswi_data) + 1), lswi_data[:, test_row, test_col], label='smoothed LSWI data', c="red")
    lswi_max = np.empty((row_num, col_num))
    lswi_min = np.empty((row_num, col_num))
    argmax = lswi_data[heading_date_arrays[test_row, test_col] - 40:heading_date_arrays[test_row, test_col] + 1, test_row, test_col].argmax()
    argmin = lswi_data[heading_date_arrays[test_row, test_col] - 40:heading_date_arrays[test_row, test_col] + 1, test_row, test_col].argmin()

    for row in range(row_num):
        for col in range(col_num):
            if not heading_date_arrays[row, col] == -1:
                lswi_max[row, col] = lswi_data[heading_date_arrays[row, col] - 40:heading_date_arrays[row, col] + 1, row, col].max()
                lswi_min[row, col] = lswi_data[heading_date_arrays[row, col] - 40:heading_date_arrays[row, col] + 1, row, col].min()

    plt.plot(argmax + heading_date_arrays[test_row, test_col] - 40 + 1, lswi_max[test_row, test_col], 'o', c="red")
    plt.plot(argmin + heading_date_arrays[test_row, test_col] - 40 + 1, lswi_min[test_row, test_col], 'o', c="red")

    rcle = np.divide((lswi_max - lswi_min), (evi2_heading - evi2_tillering))
    padding_rice_map = np.zeros((row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            if not heading_date_arrays[row, col] == -1:
                if lswi_min[row, col] > 0.1 and rcle[row, col] < 0.6:
                    padding_rice_map[row, col] = 1

    plt.legend()
    # plt.show()
    # print(heading_date_arrays)
    # print(padding_rice_map)
    return padding_rice_map


def mapping_winter_wheat():
    """
    source: Winter wheat mapping combining variations before and after estimated heading dates
    Bingwen Qiu(2017)

    :param:
      none.

    :return winter_wheat_map:
      a numpy 2d-array
      array element type: float 32
      shape: (row number, column number)
    """

    test_row = 500
    test_col = 1000
    evi2_tiff_files = get_tiff_files(FILTERED_DIR_PATH_EVI2)
    evi2_data = get_tiff_data_arrays(evi2_tiff_files, start_row=0, start_col=0, row_num=1198, col_num=1839)
    row_num = evi2_data.shape[1]
    col_num = evi2_data.shape[2]
    total_progress = row_num * col_num
    progress_bar = tqdm(range(total_progress))
    tqdm.write("start mapping winter wheat.")
    plt.plot(range(1, len(evi2_data) + 1), evi2_data[:, test_row, test_col], label='smoothed EVI2 data', c="royalblue")

    # he nan provence's average latitude and altitude
    latitude = np.zeros((row_num, col_num))
    latitude[:, :] = 33.7
    altitude = np.zeros((row_num, col_num))
    altitude[:, :] = 200

    heading_date = 2.8027 * latitude + 0.0065 * altitude + 18.5197
    early_growth_stage_length = 7.3571 * latitude + 0.0222 * altitude - 76.0539
    late_grow_stage_length = 48
    seeding_date = (heading_date - early_growth_stage_length).astype(np.int16)
    for row in range(row_num):
        for col in range(col_num):
            if seeding_date[row, col] < 1:
                seeding_date[row, col] += 361

    heading_date = heading_date.astype(np.int16)
    harvesting_date = heading_date + late_grow_stage_length
    plt.plot(heading_date[test_row, test_col] + 1, evi2_data[heading_date[test_row, test_col], test_row, test_col], 'o', c="royalblue")
    # plt.annotate(s="heading date\n(%s, %s)" % (heading_date[test_row, test_col] + 1, evi2_data[heading_date[test_row, test_col]]), xy=(heading_date[test_row, test_col] + 1, evi2_data[heading_date[test_row, test_col]]))
    plt.plot(seeding_date[test_row, test_col] + 1, evi2_data[seeding_date[test_row, test_col], test_row, test_col], 'o', c="royalblue")
    # plt.annotate(s="seeding date\n(%s, %s)" % (seeding_date[test_row, test_col] + 1, evi2_data[seeding_date[test_row, test_col]]), xy=(seeding_date[test_row, test_col] + 1, evi2_data[seeding_date[test_row, test_col]]))
    plt.plot(harvesting_date[test_row, test_col] + 1, evi2_data[harvesting_date[test_row, test_col], test_row, test_col], 'o', c="royalblue")
    # plt.annotate(s="harvesting date\n(%s, %s)" % (harvesting_date[test_row, test_col] + 1, evi2_data[harvesting_date[test_row, test_col]]), xy=(harvesting_date[test_row, test_col] + 1, evi2_data[harvesting_date[test_row, test_col]]))

    evi2_max1 = np.zeros((row_num, col_num))
    evi2_min1 = np.zeros((row_num, col_num))
    evi2_max2 = np.zeros((row_num, col_num))
    evi2_min2 = np.zeros((row_num, col_num))
    eve = np.zeros((row_num, col_num))
    evl = np.zeros((row_num, col_num))
    winter_wheat_map = np.zeros((row_num, col_num))

    for row in range(row_num):
        for col in range(col_num):
            if seeding_date[row, col] < heading_date[row, col] or harvesting_date[row, col] > 361:
                progress_bar.update(1)
                continue
            evi2_max1[row, col] = evi2_data[heading_date[row, col] - 1: seeding_date[row, col], row, col].max()
            evi2_min1[row, col] = evi2_data[heading_date[row, col] - 1: seeding_date[row, col], row, col].min()
            evi2_max2[row, col] = evi2_data[heading_date[row, col] - 1: harvesting_date[row, col], row, col].max()
            evi2_min2[row, col] = evi2_data[heading_date[row, col] - 1: harvesting_date[row, col], row, col].min()
            eve[row, col] = evi2_data[heading_date[row, col], row, col] - evi2_data[seeding_date[row, col], row, col] + evi2_max1[row, col] - evi2_min1[row, col]
            evl[row, col] = evi2_data[heading_date[row, col], row, col] - evi2_data[harvesting_date[row, col], row, col] + evi2_max2[row, col] - evi2_min2[row, col]
            if eve[row, col] > 0.3 and evl[row, col] > 0.12:
                winter_wheat_map[row, col] = 1
            progress_bar.update(1)
    progress_bar.close()
    tqdm.write("done.\n_______________________________________________________")
    plt.legend()
    # plt.show()
    return winter_wheat_map


def get_data_shape(tiff_files):
    """

    """

    return


def get_tiff_files(tiff_file_path):
    """
    read tiff file from a specified directory, and convert them into objects.

    :param tiff_file_path:
      the specified directory.

    :return tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object
    """

    tiff_file_name_list = []
    tiff_file_object_dict = {}
    for file_name in os.listdir(tiff_file_path):
        if file_name[-5:] == ".tiff":
            tiff_file_name_list.append(file_name)
    total_progress = len(tiff_file_name_list)
    tqdm.write("found " + str(total_progress) + " tiff files.")
    tqdm.write("start converting tiff files to objects.\nsource directory path: " + tiff_file_path)
    progress_bar = tqdm(range(total_progress))
    for file_name in tiff_file_name_list:
        tiff_file_object_dict[int(file_name[:-5])] = Image.open(tiff_file_path + "\\" + file_name)
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
    # tiff_data_arrays = [np.array(array)[start_row:end_row, start_col:end_col]
    #                     for array in list(tiff_file_object_dict.values())]
    tiff_data_arrays = np.empty((len(tiff_file_object_dict), row_num, col_num))
    for doy in range(1, 362):
        tiff_data_arrays[doy - 1] = np.array(tiff_file_object_dict[doy])[start_row:end_row, start_col:end_col]
    # tiff_data_arrays = np.stack(tiff_data_arrays)
    tqdm.write("done.\n_______________________________________________________")
    return tiff_data_arrays


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
            # if len(peaks) > 3:
            #     print("row: " + str(row) + " col: " + str(col) + " peaks: " + str(len(peaks)))
    # print(cropping_intensity_arrays)
    return cropping_intensity_arrays


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
    output_path = OUTPUT_PATH
    # if OUTPUT_PATH == "":
    #     output_path = TIFF_DIR_PATH + "\\" + filter_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    tqdm.write("output directory path: " + output_path)

    output_file_name = output_path + "\\" + filter_type + ".tiff"
    created_raster = driver.Create(output_file_name, col_num, row_num, 1, gdal.GDT_Float32)
    # created_raster.SetNoDataValue(INVALID_VALUE)
    # created_raster.GetRasterBand(1).SetNoDataValue(INVALID_VALUE)
    created_raster.GetRasterBand(1).WriteArray(data_array)
    tqdm.write("done.\n_______________________________________________________")
    return


if __name__ == '__main__':
    main()
