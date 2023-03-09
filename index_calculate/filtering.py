import math
import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import csv
from PIL import Image
from osgeo import gdal

XLS_DIR_PATH = "D:\\study\\yanyi\\yaogan\\week5\\20230224\\Data"
TIFF_DIR_PATH = "D:\\study\\yanyi\\yaogan\\week2\\MOD09A1\\EVI2cpy"
OUTPUT_PATH = ""
LAMBDA = 5000
WINDOW_LENGTH = 51


def main():
    # for xls_file_name, xls_file_object in get_xls_files().items():
    #     data = get_xls_data_arrays(xls_file_object)
    #     data_interpolated = interpolate_arrays(data)
    #     data_smoothed = smooth_filter_arrays(data_interpolated, "W")
    #     save_as_csv(data_smoothed, "W", xls_file_name)
    #     data_smoothed = smooth_filter_arrays(data_interpolated, "SG")
    #     save_as_csv(data_smoothed, "SG", xls_file_name)

    # plot("evi2.xlsx", 3, "evi2")
    # plot("evi2.xlsx", 4, "evi2")
    # plot("evi2.xlsx", 5, "evi2")

    data = get_tiff_data_arrays(get_tiff_files())
    print("origin data")
    print(type(data))
    print(data.shape)
    print(data)
    print("_______________")
    data_interpolated = interpolate_arrays(data)
    print(type(data_interpolated))
    print(data_interpolated.shape)
    data_smoothed = smooth_filter_arrays(data_interpolated, "W")
    print(type(data_smoothed))
    print(data_smoothed.shape)
    save_as_tiff(data_interpolated, "W")
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


def get_tiff_files():
    """
    read tiff file from a specified directory, nd convert them into objects.

    :param:
      none.

    :return tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object

    todo: extract the type of index with regular expression
    """

    tiff_file_name_list = os.listdir(TIFF_DIR_PATH)
    tiff_file_object_dict = {}
    for file_name in tiff_file_name_list:
        if file_name[-5:] == ".tiff":
            tiff_file_object_dict[file_name[13:16]] = Image.open(TIFF_DIR_PATH + "\\" + file_name)
    return tiff_file_object_dict


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
    return np.array(xls_data_arrays, dtype=object)


def get_tiff_data_arrays(tiff_file_object_dict):
    """
    convert the data in the tiff file into arrays that can be processed.

    :param tiff_file_object_dict:
      a dictionary:
        keys: tiff's DOY(string)
        values: corresponding tiff object

    :return tiff_data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.
    """

    tiff_data_arrays = [[num for row in np.array(tiff_file_object) for num in row] for tiff_file_object in
                        list(tiff_file_object_dict.values())]
    tiff_data_arrays = np.transpose(np.array(tiff_data_arrays)).tolist()
    tiff_data_arrays.insert(0, list(map(int, list(tiff_file_object_dict.keys()))))
    return np.array(tiff_data_arrays, dtype=object)


def interpolate_arrays(data_arrays):
    """
    linearly interpolate the data.

    :param data_arrays:
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

    x_interpolated = np.arange(data_arrays[0][0], data_arrays[0][len(data_arrays[0]) - 1] + 1, 1)
    interpolated_arrays = [x_interpolated]
    for y in data_arrays[1:]:
        y = y.tolist()
        x = data_arrays[0].tolist()
        index_to_delete = []
        for index, number in enumerate(y):
            if math.isnan(number):
                index_to_delete.append(index)
        for counter, index in enumerate(index_to_delete):
            index = index - counter
            x.pop(index)
            y.pop(index)

        if len(x) < 2:
            if len(x) == 1:
                interpolated_arrays.append(np.array([y[0]] * len(x_interpolated)))
            elif len(x) == 0:
                interpolated_arrays.append(np.array([0] * len(x_interpolated)))
            continue
        if 0 in index_to_delete:
            y.insert(0, y[0] + (data_arrays[0][0] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
            x.insert(0, data_arrays[0][0])
        if len(data_arrays[0]) - 1 in index_to_delete:
            y.append(y[-1] + (data_arrays[0][len(data_arrays[0]) - 1] - x[-1]) * (y[-2] - y[-1]) / (x[-2] - x[-1]))
            x.append(data_arrays[0][len(data_arrays[0]) - 1])

        f = interpolate.interp1d(np.array(x), np.array(y))
        interpolated_arrays.append(f(x_interpolated))
    return np.array(interpolated_arrays, dtype=object)


def smooth_filter_arrays(data_arrays, filter_type):
    """
    smooth filter the data.

    :param data_arrays:
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

    smoothed_arrays = [data_arrays[0]]

    # Whittaker smoother
    # source: Whittaker Smoother by Neal B. Gallagher
    if filter_type == "W":
        for y in data_arrays[1:]:
            m = len(y)
            d = np.diff(np.eye(m), 2)
            w = np.diag([int(not math.isnan(i)) for i in y])
            z = np.dot(np.linalg.inv(w + LAMBDA * np.dot(d, np.transpose(d))), (np.dot(w, y)))
            smoothed_arrays.append(z)
    # Savitzky-Golay filter
    # source: A method for reconstructing NDVI time-series based on envelope detection and the Savitzky-Golay filter
    elif filter_type == "SG":
        for y in data_arrays[1:]:
            r = signal.savgol_filter(y, WINDOW_LENGTH, 2)
            smoothed_arrays.append(r)
    return np.array(smoothed_arrays, dtype=object)


def save_as_csv(data_arrays, filter_type, xls_file_name):
    """
    save data as csv file.

    :param data_arrays:
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
        writer.writerows(data_arrays)
    return


def save_as_tiff(data_arrays, filter_type):
    """
    save data as tiff file.

    :param data_arrays:
      type: a list of numpy arrays
      the first element is the date of year(DOY),
      the other elements are the index corresponding to the day of year.

    :param filter_type:
      type: string
      the type of smoothing filter to applied to the data.
      options:
        W(Whittaker smoother) / SG(Savitzky-Golay filter)

    :return:
      none.
    """

    data_arrays = np.transpose(data_arrays)
    print("result shape:")
    print(data_arrays.shape)

    driver = gdal.GetDriverByName('GTiff')
    output_path = ""
    if OUTPUT_PATH == "":
        output_path = TIFF_DIR_PATH + "\\" + filter_type + "filtered"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # for doy, data in tiff_file_data_dict.items():
    for row in data_arrays:
        output_file_name = output_path + "\\" + str(row[0]) + ".tiff"
        created_temp = driver.Create(output_file_name, 2400, 2400, 1, gdal.GDT_Float32)
        created_temp.GetRasterBand(1).WriteArray(np.array([row[1:][i:i + 2400] for i in range(0, len(row) - 1, 2400)]))
    return


def plot(xls_file_name, row, index_type="index"):
    """
    draw a graph of the specified xls file name and the specified row number.
    the graph includes interpolated data in blue color, smoothed data.

    :param xls_file_name:
      type: string

    :param row:
      type: int

    :param index_type:
      type: string

    :return:
      none.
    """

    data = [get_xls_data_arrays(pd.read_excel(XLS_DIR_PATH + "\\" + xls_file_name, header=None))[0],
            get_xls_data_arrays(pd.read_excel(XLS_DIR_PATH + "\\" + xls_file_name, header=None))[row]]
    data_interpolated = interpolate_arrays(data)
    z = smooth_filter_arrays(data_interpolated, "W")[1]
    r = smooth_filter_arrays(data_interpolated, "SG")[1]
    x_interpolated = data_interpolated[0]
    y_interpolated = data_interpolated[1]
    plt.plot(x_interpolated, z, c='r', label='Whittaker smoother')
    plt.plot(x_interpolated, r, c='g', label='Savitzky-Golay filter')
    plt.plot(x_interpolated, y_interpolated, label='interpolated data')
    plt.plot(data[0], data[1], 'o', label='original data')
    plt.title(xls_file_name + "\nrow: " + str(row))
    plt.xlabel('DOY')
    plt.ylabel(index_type)
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
