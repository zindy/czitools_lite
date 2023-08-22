# -*- coding: utf-8 -*-

#################################################################
# File        : misc_tools.py
# Author      : sebi06
#
# Disclaimer: The code is purely experimental. Feel free to
# use it at your own risk.
#
#################################################################

from __future__ import annotations
import os
#from tkinter import filedialog
#from tkinter import *
import zarr
import dask.array as da
import numpy as np
import time
from pathlib import Path
from aicspylibczi import CziFile
import dateutil.parser as dt
from itertools import product
from czitools import metadata_tools as czimd
# from tqdm.contrib.itertools import product
from typing import List, Dict, Tuple, Optional, Type, Any, Union, Mapping


def openfile(directory: str,
             title: str = "Open CZI Image File",
             ftypename: str = "CZI Files",
             extension: str = "*.czi") -> str:
    """ Open a simple Tk dialog to select a file.

    :param directory: default directory
    :param title: title of the dialog window
    :param ftypename: name of allowed file type
    :param extension: extension of allowed file type
    :return: filepath object for the selected
    """

    # request input and output image path from user
    #root = Tk()
    #root.withdraw()
    #input_path = filedialog.askopenfile(title=title,
    #                                    initialdir=directory,
    #                                    filetypes=[(ftypename, extension)])
    #if input_path is not None:
    #    return input_path.name
    #if input_path is None:
    #    return ""
    return ""

def slicedim(array: Union[np.ndarray, da.Array, zarr.Array],
             dimindex: int,
             posdim: int) -> np.ndarray:
    """Slice out a specific dimension without (!) dropping the dimension
    of the array to conserve the dimorder string
    this should work for Numpy.Array, Dask and ZARR ...

    Example:

    - array.shape = (1, 3, 2, 5, 170, 240) and dim_order is STCZYX
    - index for C inside array = 2
    - task: Cut out the fist channel = 0
    - call: channel = slicedim(array, 0, 2)
    - the resulting channel.shape = (1, 3, 1, 5, 170, 240)

    :param array: input array
    :param dimindex: index of the slice dimension to be kept
    :param posdim: position of the dimension to be sliced
    :return: sliced array
    """

    # if posdim == 0:
    #    array_sliced = array[dimindex:dimindex + 1, ...]
    # if posdim == 1:
    #    array_sliced = array[:, dimindex:dimindex + 1, ...]
    # if posdim == 2:
    #    array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    # if posdim == 3:
    #    array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    # if posdim == 4:
    #    array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    # if posdim == 5:
    #    array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]

    idl_all = [slice(None, None, None)] * (len(array.shape) - 2)
    idl_all[posdim] = slice(dimindex, dimindex + 1, None)
    array_sliced = array[tuple(idl_all)]

    return array_sliced


def calc_scaling(data: Union[np.ndarray, da.array],
                 corr_min: float = 1.0,
                 offset_min: int = 0,
                 corr_max: float = 0.85,
                 offset_max: int = 0) -> Tuple[int, int]:
    """Calculate the scaling for better display

    :param data: Calculate min / max scaling
    :type data: Numpy.Array or dask.array or zarr.array
    :param corr_min: correction factor for minvalue, defaults to 1.0
    :type corr_min: float, optional
    :param offset_min: offset for min value, defaults to 0
    :type offset_min: int, optional
    :param corr_max: correction factor for max value, defaults to 0.85
    :type corr_max: float, optional
    :param offset_max: offset for max value, defaults to 0
    :type offset_max: int, optional
    :return: list with [minvalue, maxvalue]
    :rtype: list
    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data), np.max(data)

    end = time.time()

    minvalue = int(np.round((minvalue + offset_min) * corr_min, 0))
    maxvalue = int(np.round((maxvalue + offset_max) * corr_max, 0))

    print("Scaling:", minvalue, maxvalue)
    print("Calculation of Min-Max [s] : ", end - start)

    return minvalue, maxvalue


def sort_dict_by_key(unsorted_dict: Dict) -> Dict:
    """Sort a dictionary by key names

    Args:
        unsorted_dict: the unsorted dictionary where the keys should be sorted

    Returns:
        Dictionary with keys sorted by name
    """

    sorted_keys = sorted(unsorted_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict.update({key: unsorted_dict[key]})

    return sorted_dict


def addzeros(number: int) -> str:
    """Convert a number into a string and add leading zeros.
    Typically used to construct filenames with equal lengths.

    :param number: the number
    :type number: int
    :return: zerostring - string with leading zeros
    :rtype: str
    """

    zerostring = None

    if number < 10:
        zerostring = '0000' + str(number)
    if 10 <= number < 100:
        zerostring = '000' + str(number)
    if 100 <= number < 1000:
        zerostring = '00' + str(number)
    if 1000 <= number < 10000:
        zerostring = '0' + str(number)

    return zerostring


def get_fname_woext(filepath: Union[str, os.PathLike[str]]) -> str:
    """Get the complete path of a file without the extension
    It also works for extensions like myfile.abc.xyz
    The output will be: myfile

    :param filepath: complete filepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
    """
    # create empty string
    real_extension = ''

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remove real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def check_dimsize(mdata_entry: Union[Any, None], set2value: Any = 1) -> Union[Any, None]:
    """Check the entries for None

    Args:
        mdata_entry: entry to be checked
        set2value: value to replace None

    Returns:
        A list of dask arrays
    """

    if mdata_entry is None:
        return set2value
    if mdata_entry is not None:
        return mdata_entry


def get_planetable(czifile: Union[str, os.PathLike[str]],
                   norm_time: bool = True,
                   savetable: bool = False,
                   separator: str = ',',
                   read_one_only: bool = False,
                   index: bool = True) -> Tuple[np.recarray, Optional[str]]:
    """ Get the planetable from the individual subblocks
    Args:
        czifile: the source for the CZI image file
        norm_time: normalize the timestamps
        savetable: option save the planetable as CSV file
        separator: specify the separator for the CSV file
        read_one_only: option to read only the first entry
        index:

    Returns:
        Planetable as numpy recarray and the location of the CSV file
    """

    if isinstance(czifile, Path):
        # convert to string
        czifile = str(czifile)

    # get the czi metadata
    czi_dimensions = czimd.CziDimensions(czifile)
    aicsczi = CziFile(czifile)

    # define subblock counter
    sbcount = -1

    # check if dimensions are None (because they do not exist for that image)
    size_c = check_dimsize(czi_dimensions.SizeC, set2value=1)
    size_z = check_dimsize(czi_dimensions.SizeZ, set2value=1)
    size_t = check_dimsize(czi_dimensions.SizeT, set2value=1)
    size_s = check_dimsize(czi_dimensions.SizeS, set2value=1)
    size_m = check_dimsize(czi_dimensions.SizeM, set2value=1)

    def getsbinfo(subblock: Any) -> Tuple[float, float, float, float]:
        try:
            time = subblock.findall(".//AcquisitionTime")[0].text
            timestamp = dt.parse(time).timestamp()
        except IndexError as e:
            timestamp = 0.0

        try:
            xpos = np.double(subblock.findall(".//StageXPosition")[0].text)
        except IndexError as e:
            xpos = 0.0

        try:
            ypos = np.double(subblock.findall(".//StageYPosition")[0].text)
        except IndexError as e:
            ypos = 0.0

        try:
            zpos = np.double(subblock.findall(".//FocusPosition")[0].text)
        except IndexError as e:
            zpos = 0.0

        return timestamp, xpos, ypos, zpos

    planes = []
    # do if the data is not a mosaic
    #
    # FIXME Not exactly sure why we have these two separate loops
    # But more problematic is an exception on our Axioscan tiled images about half way through reading the bounding boxes
    # planetable, csvfile = misc_tools.get_planetable(fn, read_one_only=False, savetable=True)
    # PylibCZI_CDimCoordinatesOverspecifiedException: The coordinates are overspecified = you have specified a Dimension or Dimension value that is not valid. Z Not present in defined file Coordinates!

    if size_m > 1:

        for s, m, t, z, c in product(range(size_s),
                                     range(size_m),
                                     range(size_t),
                                     range(size_z),
                                     range(size_c)):
            sbcount += 1

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_mosaic_tile_bounding_box(S=s,
                                                            M=m,
                                                            T=t,
                                                            Z=z,
                                                            C=c)

            # read information from subblock
            sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                B=0,
                                                S=s,
                                                M=m,
                                                T=t,
                                                Z=z,
                                                C=c)

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            plane = (sbcount, s, m, t, z, c, xpos, ypos, zpos, timestamp, tilebbox.x, tilebbox.y, tilebbox.w, tilebbox.h)
            planes.append(plane)

            if read_one_only:
                break

    # do if the data is not a mosaic
    elif size_m == 1:

        for s, t, z, c in product(range(size_s),
                                  range(size_t),
                                  range(size_z),
                                  range(size_c)):
            sbcount += 1

            # get x, y, width and height for a specific tile
            tilebbox = aicsczi.get_tile_bounding_box(S=s,
                                                     T=t,
                                                     Z=z,
                                                     C=c)

            # read information from subblocks
            sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                B=0,
                                                S=s,
                                                T=t,
                                                Z=z,
                                                C=c)

            # get information from subblock
            timestamp, xpos, ypos, zpos = getsbinfo(sb)

            plane = (sbcount, s, 0, t, z, c, xpos, ypos, zpos, timestamp, tilebbox.x, tilebbox.y, tilebbox.w, tilebbox.h)
            planes.append(plane)

            if read_one_only:
                break

    # cast data  types
    types = {'Subblock':np.int32,
            'Scene':np.int32,
            'Tile':np.int32,
            'T':np.int32,
            'Z':np.int32,
            'C':np.int16,
            'X[micron]':float,
            'Y[micron]':float,
            'Z[micron]':float,
            'Time[s]':float,
            'xstart':np.int32,
            'ystart':np.int32,
            'width':np.int32,
            'height':np.int32}

    arr_czi = np.array(planes,dtype=list(types.items()))

    # normalize time stamps
    if norm_time:
        arr_czi = norm_columns(arr_czi, colname='Time[s]', mode='min')

    # save planetable as CSV file
    if savetable:
        csvfile = save_planetable(arr_czi, czifile, separator=separator, index=index)
    if not savetable:
        csvfile = None

    return arr_czi, csvfile


def norm_columns(arr: np.recarray,
                 colname: str = 'Time [s]',
                 mode: str = 'min') -> np.recarray:
    """Normalize a specific column inside a numpy record array
    Args:
        arr: numpy record array
        colname: Name of the column to be normalized, defaults to 'Time [s]'
        mode: Mode of Normalization, defaults to 'min'

    Returns:
        numpy record array with normalized columns
    """

    # normalize columns according to min or max value
    if mode == 'min':
        min_value = arr[colname].min()
        arr[colname] = arr[colname] - min_value

    if mode == 'max':
        max_value = arr[colname].max()
        arr[colname] = arr[colname] - max_value

    return arr


def filter_planetable(arr: np.recarray,
                      s: int = 0,
                      t: int = 0,
                      z: int = 0,
                      c: int = 0) -> np.recarray:
    """Filter the planetable for specific dimension entries
    Args:
        planetable: The planetable to be filtered
        s: scene index
        t: time index
        z: z-plane index
        c: channel index

    Returns:
        The filtered planetable
    """

    # filter planetable for specific scene
    if s > planetable['Scene'].max():
        print('Scene Index was invalid. Using Scene = 0.')
        s = 0
    pt = planetable[planetable['Scene'] == s]

    # filter planetable for specific timepoint
    if t > planetable['T'].max():
        print('Time Index was invalid. Using T = 0.')
        t = 0
    pt = planetable[planetable['T'] == t]

    # filter resulting planetable pt for a specific z-plane
    try:
        if z > planetable['Z[micron]'].max():
            print('Z-Plane Index was invalid. Using Z = 0.')
            zplane = 0
            pt = pt[pt['Z[micron]'] == z]
    except KeyError as e:
        if z > planetable['Z [micron]'].max():
            print('Z-Plane Index was invalid. Using Z = 0.')
            zplane = 0
            pt = pt[pt['Z [micron]'] == z]

    # filter planetable for specific channel
    if c > planetable['C'].max():
        print('Channel Index was invalid. Using C = 0.')
        c = 0
    pt = planetable[planetable['C'] == c]

    # return filtered planetable
    return pt


def save_planetable(arr:np.recarray,
                    filename: str,
                    separator: str = ',',
                    index: bool = True) -> str:
    """Saves a numpy record array as a CSV file.

    Args:
        arr (numpy record array): The array to be saved as CSV.
        filename (str): The filename of the CSV file to be written.
        separator (str, optional): The separator character for the CSV file. Defaults to ','.
        index (bool, optional): Whether to include the index in the CSV file. Defaults to True.

    Returns:
        str: The filename of the CSV file that was written.
    """
    # Generate the filename for the planetable CSV.
    csvfile = os.path.splitext(filename)[0] + '_planetable.csv'

    # Here we generate the column names
    fields = arr.dtype.names

    if index == False:
        #Remove the Subblock from the array view
        arr = arr[list(arr.dtype.names)[1:]]

    np.savetxt(csvfile, arr, delimiter=separator)

    return csvfile


def expand5d(array: np.ndarray) -> np.ndarray:
    """Expands a multi-dimensional numpy array to 5 dimensions.

    Args:
        array (np.ndarray): The numpy array to be extended to 5 dimensions.

    Returns:
        np.ndarray: The 5-dimensional numpy array.
    """
    # Expand the input array along the third-to-last dimension.
    array = np.expand_dims(array, axis=-3)
    # Expand the result along the fourth-to-last dimension.
    array = np.expand_dims(array, axis=-4)
    # Expand the result along the fifth-to-last dimension.
    array5d = np.expand_dims(array, axis=-5)

    return array5d
