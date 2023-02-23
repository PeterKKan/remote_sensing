import numpy as np
from pyhdf.SD import SD, SDC
from osgeo import gdal
from osgeo import osr
import os

HDF_DIR_PATH = ""  # path of hdf file directory
OUTPUT_PATH = ""  # path of the output directory
R = "sur_refl_b01"  # band name and its corresponding SDS name
NIR = "sur_refl_b02"
B = "sur_refl_b03"
SWIR = "sur_refl_b06"


def main():
    calculate_index(get_hdf_files())


def get_hdf_files():
    """
    read hdf files from a specified directory and
    convert them into python hdf objects.

    :param:
      none.

    :return hdf_file_object_dict:
      a dictionary:
        keys: hdf file name without ".hdf"
        values: corresponding python hdf object
    """

    hdf_file_name_list = os.listdir(HDF_DIR_PATH)
    hdf_file_object_dict = {}
    for file_name in hdf_file_name_list:
        if file_name[-4:] == ".hdf":
            hdf_file_object_dict[file_name[:-4]] = SD(HDF_DIR_PATH + "\\" + file_name, SDC.READ)
    return hdf_file_object_dict


def calculate_index(hdf_file_object_dict):
    """
    calculate NDVI, EVI, EVI2, LSWI of each python hdf object and
    save the result to a specified directory.

    :param hdf_file_object_dict:
      a dictionary:
        keys: hdf file name without ".hdf"
        values: corresponding python hdf object

    :return:
      none.
    """

    for hdf_file_name, hdf_object in hdf_file_object_dict.items():
        r = get_band_data_matrix(hdf_object, R)
        n = get_band_data_matrix(hdf_object, NIR)
        b = get_band_data_matrix(hdf_object, B)
        swir = get_band_data_matrix(hdf_object, SWIR)

        # source: Development of a two-band enhanced vegetation index without a blue band
        ndvi = reasonable_divide((n - r), (n + r))
        # source: Development of a two-band enhanced vegetation index without a blue band
        evi = 2.5 * reasonable_divide((n - r), (n + 6 * r - 7.5 * b + 1))
        # source: Development of a two-band enhanced vegetation index without a blue band
        evi2 = 2.5 * reasonable_divide((n - r), (n + 2.4 * r + 1))
        # source: Mapping paddy rice agriculture in southern China using multi-temporal MODIS images
        lswi = reasonable_divide((n - swir), (n + swir))

        save_as_tiff(ndvi, "NDVI", hdf_file_name)
        save_as_tiff(evi, "EVI", hdf_file_name)
        save_as_tiff(evi2, "EVI2", hdf_file_name)
        save_as_tiff(lswi, "LSWI", hdf_file_name)


def get_band_data_matrix(hdf_object, band):
    """
    get data of a band, and convert it into numpy matrix.

    :param hdf_object:
      the python hdf object.

    :param band:
      which band's data need to convert.
      options:
         R(red)/NIR/B(blue)/SWIR

    :return:
      a numpy matrix of data of a band.
    """
    band_object = hdf_object.select(band)
    band_data = band_object.get()
    return np.matrix(band_data)


def reasonable_divide(divisor, dividend):
    """
    a division that can avoid errors like divide by zero.

    :param divisor:
      type: numpy matrix

    :param dividend:
      type: numpy matrix

    :return:
      division result.
      type: numpy matrix
    """
    return np.divide(divisor, dividend, out=np.zeros_like(divisor, dtype=np.float64), where=dividend != 0)


def save_as_tiff(matrix, index_type, hdf_file_name):
    """
    save the calculated matrix as a tiff file.

    :param matrix:
      matrix that needs to be written to the tiff file.

    :param index_type:
      the index type.

    :param hdf_file_name:
      hdf file name without ".hdf"

    :return:
      none.
    """

    x_pixels = matrix.shape[1]  # number of pixels in x
    y_pixels = matrix.shape[0]  # number of pixels in y
    driver = gdal.GetDriverByName('GTiff')

    output_path = ""
    if OUTPUT_PATH == "":
        output_path = HDF_DIR_PATH + "\\" + index_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file_name = output_path + "\\" + hdf_file_name[0: -13] + index_type + ".tiff"
    created_temp = driver.Create(output_file_name, x_pixels, y_pixels, 1, gdal.GDT_Float32)
    created_temp.GetRasterBand(1).WriteArray(matrix)


if __name__ == '__main__':
    main()
