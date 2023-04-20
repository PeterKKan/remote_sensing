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

FILTERED_DIR_PATH_EVI2 = "D:\\study\\yanyi\\yaogan\\week2\\MOD09A1\\EVI2\\Wfiltered"
FILTERED_DIR_PATH_LSWI = ""


def main():
    mapping_paddy_rice()
    return


def mapping_paddy_rice():
    test_row = 0
    test_col = 0
    evi2_tiff_files = get_tiff_files(FILTERED_DIR_PATH_EVI2)
    evi2_data = get_tiff_data_arrays(evi2_tiff_files, start_row=0, start_col=0, row_num=50, col_num=50)
    print(evi2_tiff_files)
    plt.plot(range(1, len(evi2_data) + 1), evi2_data[0:, test_row, test_col], label='smoothed data')
    row_num = evi2_data.shape[1]
    col_num = evi2_data.shape[2]
    heading_date_arrays = np.empty((row_num, col_num))
    for row in range(row_num):
        for col in range(col_num):
            data = evi2_data[:, row, col]
            peaks, peak_heights = signal.find_peaks(data, height=0.3, distance=60, prominence=0.1)
            if len(peaks) >= 1:
                heading_date_arrays[row, col] = peaks[0]
            else:
                heading_date_arrays[row, col] = -1
    plt.legend()
    plt.show()
    print(heading_date_arrays)
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
        tiff_file_object_dict[file_name[:-5]] = Image.open(tiff_file_path + "\\" + file_name)
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


if __name__ == '__main__':
    main()
