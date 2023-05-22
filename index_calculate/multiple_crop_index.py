import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from osgeo import gdal

TIFF_DIR_PATH = "D:\\BaiduNetdiskDownload\\EVI2"
OUTPUT_PATH = ""


def main():
    data = get_tiff_data_arrays(get_tiff_files(TIFF_DIR_PATH), start_row=0, start_col=0, row_num=1198, col_num=1839)
    ci = calculate_multi_crop_index(data)
    save_as_tiff_2d(ci, "ci")
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

    print(len(tiff_file_object_dict))
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


def calculate_multi_crop_index(data_arrays):
    row_num = data_arrays.shape[1]
    col_num = data_arrays.shape[2]

    diff_arrays = np.empty((data_arrays.shape[0] - 1, row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            for height in range(data_arrays.shape[0] - 1):
                if data_arrays[height, row, col] < data_arrays[height + 1, row, col]:
                    diff_arrays[height, row, col] = 1
                elif data_arrays[height, row, col] == data_arrays[height + 1, row, col]:
                    diff_arrays[height, row, col] = 0
                elif data_arrays[height, row, col] > data_arrays[height + 1, row, col]:
                    diff_arrays[height, row, col] = -1

    peak_arrays = np.empty((diff_arrays.shape[0] - 1, row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            for height in range(diff_arrays.shape[0] - 1):
                if diff_arrays[height + 1, row, col] - diff_arrays[height, row, col] == -1:
                    peak_arrays[height, row, col] = data_arrays[height, row, col]
                elif diff_arrays[height + 1, row, col] - diff_arrays[height, row, col] == -2:
                    peak_arrays[height, row, col] = data_arrays[height, row, col]
                else:
                    peak_arrays[height, row, col] = 0

    multi_crop = np.zeros((row_num, col_num))

    for row in range(row_num):
        for col in range(col_num):
            ind_peak = [index for (index, value) in enumerate(peak_arrays[:, row, col]) if value > 0.3]
            siz_peak = len(ind_peak)
            if siz_peak == 1:
                multi_crop[row, col] = 1
            elif siz_peak == 2:
                if (ind_peak[1] - ind_peak[0]) >= 5*16:
                    multi_crop[row, col] = 2
                else:
                    multi_crop[row, col] = 1
    return multi_crop


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
    # created_raster.GetRasterBand(1).SetNoDataValue(INVALID_VALUE)
    created_raster.GetRasterBand(1).WriteArray(data_array)
    tqdm.write("done.\n_______________________________________________________")
    return


if __name__ == '__main__':
    main()
