import numpy as np
from pyhdf.SD import SD, SDC
from osgeo import gdal
from osgeo import osr
import os

HDF_DIR_PATH = "D:\\study\\yanyi\\yaogan\\week2\\MOD09A1"  # path of hdf file directory
OUTPUT_PATH = ""  # path of the output directory
R = "sur_refl_b01"  # band name and its corresponding SDS name
NIR = "sur_refl_b02"
B = "sur_refl_b03"
SWIR = "sur_refl_b06"
QC = "sur_refl_qc_500m"  # 500m Reflectance Band Quality


def main():
    calculate_index(get_hdf_files())
    return


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


def calculate_index(hdf_file_object_dict, scale_factor=0.0001, offset=0):
    """
    calculate NDVI, EVI, EVI2, LSWI of each python hdf object and
    save the result to a specified directory.

    :param hdf_file_object_dict:
      a dictionary:
        keys: hdf file name without ".hdf"
        values: corresponding python hdf object

    :param scale_factor:
      scale factor of the science dataset.
      type: float

    :param offset:
      offset of the science dataset.
      type: float

    :return:
      none.
    """

    for hdf_file_name, hdf_object in hdf_file_object_dict.items():
        r = scale_factor * get_band_data_matrix(hdf_object, R) + offset
        n = scale_factor * get_band_data_matrix(hdf_object, NIR) + offset
        b = scale_factor * get_band_data_matrix(hdf_object, B) + offset
        swir = scale_factor * get_band_data_matrix(hdf_object, SWIR) + offset
        qc = get_quality_control_mask(hdf_object)

        # source: Development of a two-band enhanced vegetation index without a blue band
        ndvi = np.where(qc == 1, reasonable_divide((n - r), (n + r)), np.nan)
        # source: Development of a two-band enhanced vegetation index without a blue band
        evi = np.where(qc == 1, 2.5 * reasonable_divide((n - r), (n + 6 * r - 7.5 * b + 1)), np.nan)
        # source: Development of a two-band enhanced vegetation index without a blue band
        evi2 = np.where(qc == 1, 2.5 * reasonable_divide((n - r), (n + 2.4 * r + 1)), np.nan)
        # source: Mapping paddy rice agriculture in southern China using multi-temporal MODIS images
        lswi = np.where(qc == 1, reasonable_divide((n - swir), (n + swir)), np.nan)

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
         R(red)/NIR/B(blue)/SWIR/QC

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


def get_quality_control_mask(hdf_object):
    """
    get data of band quality matrix,
    and convert it into mask matrix.

    :param hdf_object:
      the python hdf object.

    :return quality_control_mask:
      quality control mask.
      type: numpy matrix.
    """

    band_quality_matrix = get_band_data_matrix(hdf_object, QC).tolist()
    quality_control_mask = []
    for row in band_quality_matrix:
        quality_control_mask.append([int(bin(num)[-2:] in ['00']) for num in row])
    return np.matrix(quality_control_mask)


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

    todo: return a bool value and catch the error.
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


def get_SDS_info(hdf_object, flag=0):
    """
    get the number and name of SDS.

    :param hdf_object:
      the python hdf object.

    :param flag:
      type: int
      when it is set to 0, print out SDS_info.

    :return SDS_info:
      a list:
        the first element is the number of SDS.
        the second element is a list of all SDS name(string).
    """

    sds_info = [hdf_object.info()[0], list(hdf_object.datasets().keys())]
    if flag == 0:
        print(sds_info)
    return sds_info


if __name__ == '__main__':
    main()
