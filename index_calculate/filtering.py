import math
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import csv


XLS_DIR_PATH = "D:\\study\\yanyi\\yaogan\\week5\\20230224\\Data"
OUTPUT_PATH = ""
LAMBDA = 10000
WINDOW_LENGTH = 51


def main():
    for xls_file_name, xls_file_object in get_xls_files().items():
        data = get_xls_data_arrays(xls_file_object)
        data_interpolated = interpolate_arrays(data)
        data_smoothed = smooth_filter_arrays(data_interpolated, "W")
        save_as_csv(data_smoothed, "W", xls_file_name)
        data_smoothed = smooth_filter_arrays(data_interpolated, "SG")
        save_as_csv(data_smoothed, "SG", xls_file_name)
    return


def get_xls_files():
    """
    read xls(x) file from a specified directory,
    and convert them into xls file objects(pandas data frame).

    :param:
      none.

    :return xls_file_object_dict:
      a dictionary:
        keys: xls(x) file name without ".xls(x)"
        values: corresponding xls file object

    todo: get single xls file mode.
    """

    xls_file_name_list = os.listdir(XLS_DIR_PATH)
    xls_file_object_dict = {}
    for file_name in xls_file_name_list:
        if file_name[-4:] == ".xls":
            xls_file_object_dict[file_name[:-4]] = pd.read_excel(XLS_DIR_PATH + "\\" + file_name, header=None)
        elif file_name[-5:] == ".xlsx":
            xls_file_object_dict[file_name[:-5]] = pd.read_excel(XLS_DIR_PATH + "\\" + file_name, header=None)
    return xls_file_object_dict


def get_xls_data_arrays(xls_file_object):
    """
    get data from the xls file object and convert them into arrays,
    also remove empty data.

    :param xls_file_object:
      the source of the data.

    :return xls_data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    todo: select the range of rows and columns you want to get.
    """

    xls_data_arrays = xls_file_object.iloc[0:, 1:].values.tolist()  # iloc[row, col]
    return np.array(xls_data_arrays)


def interpolate_arrays(xls_data_arrays):
    """
    linearly interpolate the data.

    :param xls_data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    :return interpolated_arrays:
      interpolated data.
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    todo: select the range of interpolation and the step size of interpolation.
          apply other interpolation methods.
    """

    x_interpolated = np.arange(1, 362, 1)
    interpolated_arrays = [x_interpolated]
    for y in xls_data_arrays[1:]:
        y = y.tolist()
        x = xls_data_arrays[0].tolist()
        for index, number in enumerate(y):
            if math.isnan(number):
                y.pop(index)
                x.pop(index)
        f = interpolate.interp1d(np.array(x), np.array(y))
        interpolated_arrays.append(f(x_interpolated))
    return interpolated_arrays


def smooth_filter_arrays(xls_data_arrays, filter_type):
    """
    smooth filter the data.

    :param xls_data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    :param filter_type:
      type: string
      the type of smoothing filter to apply to the data.
      options:
        W(Whittaker smoother) / SG(Savitzky-Golay filter)

    :return smoothed_arrays:
      smoothed data.
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.
    """

    smoothed_arrays = [xls_data_arrays[0]]

    # Whittaker smoother
    # source: Whittaker Smoother by Neal B. Gallagher
    if filter_type == "W":
        for y in xls_data_arrays[1:]:
            m = len(y)
            d = np.diff(np.eye(m), 2)
            w = np.diag([int(not math.isnan(i)) for i in y])
            z = np.dot(np.linalg.inv(w + LAMBDA * np.dot(d, np.transpose(d))), (np.dot(w, y)))
            smoothed_arrays.append(z)
    # Savitzky-Golay filter
    # source: A method for reconstructing NDVI time-series based on envelope detection and the Savitzky-Golay filter
    elif filter_type == "SG":
        for y in xls_data_arrays[1:]:
            r = signal.savgol_filter(y, WINDOW_LENGTH, 2)
            smoothed_arrays.append(r)
    return smoothed_arrays


def save_as_csv(xls_data_arrays, filter_type, xls_file_name):
    """
    save data as csv file.

    :param xls_data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    :param filter_type:
      type: string
      the type of smoothing filter to applied to the data.
      options:
        W(Whittaker smoother) / SG(Savitzky-Golay filter)

    :param xls_file_name:
      original xls(x) file name without ".xls(x)"

    :return:
      none.

    todo: return a bool value and catch the error.
          convert numpy array to list.
    """

    output_path = ""
    if OUTPUT_PATH == "":
        output_path = XLS_DIR_PATH + "\\"
    with open(output_path + xls_file_name + filter_type + ".csv", 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerows(xls_data_arrays)
    return


if __name__ == '__main__':
    main()
