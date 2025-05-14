"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Code by Atchuth Naveen
Updated and maintained by Andy Thai
Code developed at UC Irvine.

Corrects deformation in individual images and stitches them into a large mosaic section. 
"""

# Standard library imports
import sys
import argparse
import os
import time
import itertools
import sys
import logging
logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)

# Third party imports
import glob
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from six import iteritems

# Third party imports - image and array processing
import cv2
import SimpleITK as sitk
import numpy as np
import skimage
from skimage.transform import resize
import skimage.morphology
from scipy.special import binom

# Custom imports
from stitcher import Stitcher
from tile import Tile


def bernstein(u, n: int, k: int) -> float:
    """Bernstein polynomial for deformation mapping.

    Args:
        u (_type_): Input value
        n (int): Top binomial coefficient
        k (int): Bottom binomial coefficient

    Returns:
        float: Bernstein polynomial output
    """
    return binom(n, k) * u**k * (1 - u)**(n - k)


def barray(u: np.ndarray, v: np.ndarray, n: int, m: int) -> np.ndarray:
    """Generates a Bernstein polynomial matrix.

    Args:
        u (np.ndarray): _description_
        v (np.ndarray): _description_
        n (int): _description_
        m (int): _description_

    Returns:
        np.ndarray: Output matrix
    """
    bmatrix = np.zeros((len(u), (n + 1) * (m + 1)))
    for i in range(n + 1):
        for j in range(m + 1):
            bmatrix[:, i * (m + 1) + j] = bernstein(u, n, i) * bernstein(v, m, j)
            #blist.append(bernstein(u,n,i)* bernstein(v,m,j))
    return bmatrix


def create_perfect_grid(nhs: int, nvs: int, lw: float, sw: float) -> np.ndarray:
    """Generates a perfect grid image.

    Args:
        nhs (int): number of horizontal squares
        nvs (int): number of vertical squares
        lw (float): line width
        sw (float): square width

    Returns:
        np.ndarray: Grid image
    """
    xs = 20
    ys = 20
    
    im = np.zeros((2 * xs + nvs * sw + lw, 2 * ys + nhs * sw + lw))
    # Generate horizontal lines
    for i in range(nhs + 1):
        cv2.line(im, (xs + i * sw, ys), (xs + i * sw, im.shape[0] - ys - int(lw / 2)), (255, 255, 255), thickness=lw)
    # Generate vertical lines
    for i in range(nvs + 1):
        cv2.line(im, (xs, ys + i * sw), (im.shape[1] - xs - int(lw / 2), ys + i * sw), (255, 255, 255), thickness=lw) 
    return im


def get_deformation_map(width: int, height: int, kx, ky) -> tuple:
    """Retrieves deformation map for image correction.

    Args:
        width (int): Width of map
        height (int): Height of map
        kx (_type_): Bezier patch parameters
        ky (_type_): Bezier patch parameters

    Returns:
        tuple: _description_
    """
    px = []
    py = []
    for i in range(2 * width):
        for j in range(2 * height):
            px.append(j)
            py.append(i)
    px = np.asarray(px)
    py = np.asarray(py)
    px_ = np.asarray(px / (2 * float(height)))
    py_ = np.asarray(py / (2 * float(width)))
    
    pX_ = np.matmul(barray(px_, py_, 4, 4), kx)
    pY_ = np.matmul(barray(px_, py_, 4, 4), ky)
    pX_ = pX_ * height
    pY_ = pY_ * width
    # Clip values
    pX_[pX_ <= 0] = 0
    pX_[pX_ >= height - 1] = height - 1
    pY_[pY_ <= 0] = 0
    pY_[pY_ >= width - 1] = width -1
    
    return pX_, pY_


def correct_deformation(im0: np.ndarray, H, pX_, pY_) -> np.ndarray:
    """_summary_

    Args:
        im0 (np.ndarray): Image array
        H (_type_): Homography information
        pX_ (_type_): _description_
        pY_ (_type_): _description_

    Returns:
        np.ndarray: Deformation corrected image
    """
    im_warp = cv2.warpPerspective(im0, H, (im0.shape[1], im0.shape[0]))
    im_warp = im_warp[20:794, 20:776]
    h,w  = im_warp.shape
    
    im = np.zeros(pX_.shape).astype(np.float64)
    x1 = np.floor(pX_).astype(int)
    x2 = np.ceil(pX_).astype(int)
    y1 = np.floor(pY_).astype(int)
    y2 = np.ceil(pY_).astype(int)
    
    dx1 = pX_ - x1
    dx2 = x2 - pX_
    dy1 = pY_ - y1
    dy2 = y2 - pY_
    dx1[np.where(y1 == y2)] = 1
    dy1[np.where(x1 == x2)] = 1
    
    im_warp1d = im_warp.ravel()
    im = im_warp1d[y1 * w + x1] * dx1 * dy1 + \
         im_warp1d[y1 * w + x2] * dx2 * dy1 + \
         im_warp1d[y2 * w+ x1] * dy2 * dx1 + \
         im_warp1d[y2 * w + x2] * dy2 * dx2
    
    im = np.reshape(im, (2 * h, 2 * w))
    im = resize(im, (h, w), preserve_range=True)
    return im


def get_missing_tile_paths(missing_tiles) -> list:
    """_summary_

    Args:
        missing_tiles (_type_): _description_

    Returns:
        list: Missing tile paths
    """
    paths = []

    for index, path in iteritems(missing_tiles):
        spath = ','.join(map(str, path))
        logging.info('Writing missing tile path for tile {0} as {1}'.format(index, spath))
        paths.append(spath)

    return paths


def read_image(file_name: str) -> np.ndarray:
    """Reads image from filepath string.

    Args:
        file_name (str): Input file path

    Returns:
        np.ndarray: Image array
    """
    image = sitk.ReadImage(str(file_name))
    return sitk.GetArrayFromImage(image)
    #return np.flipud(sitk.GetArrayFromImage(image)).T


def write_output(imgarr: np.ndarray, path: str):
    """Writes image array to file.

    Args:
        imgarr (np.ndarray): Image array
        path (str): Output file path
    """
    imgarr[imgarr < 0] = 0
    image = sitk.GetImageFromArray(imgarr)
    image = sitk.Cast(image, sitk.sitkUInt16)
    sitk.WriteImage(image, path)


def normalize_image_by_median(image: np.ndarray) -> np.ndarray:
    """Normalizes image by median value.

    Args:
        image (np.ndarray): Input image array

    Returns:
        np.ndarray: Normalized image array
    """
    median = np.median(image)

    if median != 0:
        image = np.divide(median, image)
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0

    return image


def load_average_tile(path: str) -> np.ndarray:
    """Loads average tile from file and normalizes it by median.

    Args:
        path (str): File path

    Returns:
        np.ndarray: Normalized average tile
    """
    tile = read_image(path)
    return normalize_image_by_median(tile)


def get_section_avg(tiles: list, median_thresh: float = 20.0) -> list:
    """Calculates average tiles for each channel using each section.

    Args:
        tiles (list): Tile information

    Returns:
        list: A list of average tiles for each channel
    """
    #logging.info("tiles:", tiles)
    imlist = [[], [], [], []]
    avg_tiles = []
    for tile in tiles:
        try:
            im = read_image(tile['path'])
            #im = cv2.resize(im, (832,832))
            if np.median(im) >= median_thresh:
                imlist[tile["channel"] - 1].append(im)
        except(IOError, OSError, RuntimeError) as err:
            logging.info('Did not find image tile for channel {0} (zero-indexed)'.format(tile["channel"] - 1))
    
    for i in range(0, 4):
        if len(imlist[i]) == 0:
            imlist[i].append(np.ones((832, 832)))
    
    avg_tiles.append(np.mean(imlist[0], axis=0))
    avg_tiles.append(np.mean(imlist[1], axis=0))
    avg_tiles.append(np.mean(imlist[2], axis=0))
    avg_tiles.append(np.mean(imlist[3], axis=0))
    return avg_tiles


def generate_avg_tiles(section_jsons: list, avg_tiles_dir: str, n_threads: int, median_thresh: float = 20.0):
    """Generates average tiles for each channel.

    Args:
        section_jsons (list): Data for each section
        avg_tiles_dir (str): File path for average tiles
        n_threads (int): Number of threads to run the section averaging
    """
    always_regenerate = True
    if not os.path.isdir(avg_tiles_dir) or always_regenerate:
        os.mkdir(avg_tiles_dir)
        logging.info('Generating average tiles...')

        imlist = Parallel(n_jobs=n_threads, verbose=13)(delayed(get_section_avg)(section_json['tiles'], median_thresh) 
                                                        for section_json in section_jsons)
        imlist = np.asarray(imlist)
        
        avg_tiles = np.mean(imlist, axis=0)
        for i, im in enumerate(avg_tiles):
            image = sitk.GetImageFromArray(im)
            image = sitk.Cast(image, sitk.sitkFloat32)
            sitk.WriteImage(image, os.path.join(avg_tiles_dir, "avg_tile_" + str(i) + ".tif"))
    else:
        logging.info('Average tiles already exist. Skipping generation...')


def preprocess(img: np.ndarray, min_val: float = None, max_val: float = 400) -> np.ndarray:
    """Preprocesses volume data. Clips maximum value at max_val and then normalizes volume
    between 0-255.

    Args:
        img (np.ndarray): Input image array
        min_val (float, optional): Minimum value to clip image values to. Defaults to None.
        max_val (float, optional): Maximum value to clip image values to. Defaults to 400.

    Returns:
        np.ndarray: Clipped and preprocessed image array
    """
    data = img.copy()
    data[data > max_val] = max_val
    if min_val:
        data[data < min_val] = min_val
    data = cv2.normalize(src=data, dst=None, alpha=0, beta=255, 
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return data


def generate_mask(im: np.array, thresh: int = 15, 
                  gauss_kernel: int = 55, min_size: int = 64, area_threshold: int = 64) -> np.ndarray:
    """Generates mask used to determine which pixels to apply average tile brightness correction to. 

    Args:
        im (np.array): Input image array
        thresh (int, optional): Threshold to determine which is tissue and which is background. Defaults to 15.
        gauss_kernel (int, optional): How big the Gaussian kernel is. Defaults to 55.
        min_size (int, optional): The minimum size of a speck for it to not be removed. Defaults to 64.
        area_threshold (int, optional): The minimum size of a hole for it not to be filled in. Defaults to 64.

    Returns:
        np.ndarray: _description_
    """
    y, x = im.shape
    # Settings
    min_size = max(min_size, 
                   int(max(y, x) * 0.20))
    area_threshold = max(area_threshold, 
                         int(max(y, x) * 20))

    # Preprocess image
    mask = im.copy()
    mask = preprocess(mask, min_val=0, max_val=400)
    mask = cv2.GaussianBlur(mask, (gauss_kernel, gauss_kernel), 0)

    # Threshold it
    mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1].astype(bool)
    mask = np.where(mask > 0, 1, 0).astype(bool)

    # Remove small specks from background
    mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=min_size).astype(bool)
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=area_threshold).astype(bool)

    return mask


def generate_tiles(tiles: list, avg_tiles: list, 
                   H, pX_, pY_, thresh: int = 15, ch: int = None, 
                   save_undistorted: bool = False, median_thresh: float = None):
    """Generate the images for the tiles and applies processing on them.

    Args:
        tiles (list): Tile information
        avg_tiles (list): List of average tiles
        output_dir (str): Where to save the undistorted tiles
        H (_type_): Homography information
        pX_ (_type_): _description_
        pY_ (_type_): _description_
        ch (int, optional): Which channel to generate for. If a channel is provided, only generates for that channel, 
                            otherwise generates for all of them if None. Defaults to None.
        save_undistorted (bool, optional): If True, saves the images without distortion correction. Defaults to False.
        median_thresh (float, optional): Threshold for median value to binarize image. Defaults to None.

    Yields:
        Iterator[list]: Processed tile objects
    """    
    # Remove irrelevant channels if a specific channel is provided
    if ch is not None:
        new_tiles = []
        for t in tiles:
            if t['channel'] == ch + 1:
                new_tiles.append(t)
        tiles = new_tiles
    
    for tile_params in tiles:
        tile = tile_params.copy()
        try:
            im = read_image(tile['path'])
            #im = cv2.resize(im, (832,832))
            
            ###### PROCESSING ######
            y, x = im.shape

            # Settings to generate mask
            mask = generate_mask(im, thresh=thresh)

            # Apply mask to image
            im_masked = np.where(mask, im.copy(), 0)
            im_masked = np.multiply(im_masked, avg_tiles[tile['channel'] - 1])
            im[mask != 0] = im_masked[mask != 0]

            # Original stitching, comment the last 4 lines and uncomment the below to use
            #im = np.multiply(im, avg_tiles[tile['channel'] - 1])
            
            # Correct deformation
            im_corrected = correct_deformation(im, H, pX_, pY_)
            tile['image'] = im_corrected
            if save_undistorted:
                undistorted_tile_path = os.path.join(undistorted_dir, 
                                                     "ch{}".format(tile['channel'] - 1), 
                                                     os.path.split(tile['path'])[1])
                write_output(np.ascontiguousarray(im_corrected), undistorted_tile_path)
            tile['is_missing'] = False
            
        # If the tile is missing, set the image to None and is_missing to True
        except (IOError, OSError, RuntimeError) as err:
            tile['image'] = None
            tile['is_missing'] = True
            
        # Binarize tile image if median thresh is provided for visualization purposes in preview
        if median_thresh is not None:
            tile_median = np.median(tile['image'])
            if tile_median >= median_thresh:
                tile['image'] = np.ones(tile['image'].shape)
            else:
                tile['image'] = np.zeros(tile['image'].shape)
        
        # Decrement channel by 1 to make it zero-indexed and yield the tile object
        tile['channel'] = tile['channel'] - 1
        tile_obj = Tile(**tile)
        del tile
        yield tile_obj


def create_section_json(sno: int, sectionName: str, mosaic_data: list, depth: int = 0):
    """Creates a JSON object for storing section information.

    Args:
        sno (int): Section index number
        sectionName (str): Name of the section
        mosaic_data (list): Mosaic data information
        depth (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: Section JSON data
    """
    tyx = -3
    tyy = -43
    txx = -25
    txy = 5
    margins = {"row": 0, "column": 0}
    size = {"row": 774, "column": 756}
    startx = 200
    starty = 200

    mcolumns = int(mosaic_data["mcolumns"])
    mrows = int(mosaic_data["mrows"])
    index_ = (sno * int(mosaic_data["layers"]) + depth) * mrows * mcolumns 
    image_dimensions = {"row": mrows * size['row'] + 2 * startx, 
                        "column": mcolumns * size['column'] + 2 * starty}
    
    section_json = {}
    section_json["mosaic_parameters"] = mosaic_data
    tiles = []
    for ncol in range(mcolumns):
        for nrow in range(mrows):
            index = index_ + (ncol) * mrows + nrow
            #index = sno*mrows*mcolumns + (ncol)*mrows+nrow
            tile_paths = glob.glob(f'{sectionName}/*-{index}_*.tif')
            #print(f'{sectionName}/*-{index}_*.tif', tile_paths, sectionName, index)
            if len(tile_paths) == 0:
                continue
            bounds = {}
            row = {}
            col = {}
            if ncol % 2 == 0:
                row["start"] = starty + nrow * size["row"] + nrow * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx + nrow * tyx 
                col["end"] = col["start"] + size["column"]
            else:
                row["start"] = starty + (mrows - nrow - 1) * size["row"] + (mrows - nrow - 1) * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx + (mrows - nrow - 1) * tyx 
                col["end"] = col["start"] + size["column"]
            bounds["row"] = row
            bounds["column"] = col
            for ch, path in enumerate(tile_paths):
                tile_data = {}
                tile_data["path"] = path
                tile_data["bounds"] = bounds
                tile_data["margins"]= margins
                tile_data["size"] = size
                tile_data["channel"] = ch + 1
                tile_data["index"] = index
                tiles.append(tile_data)
        #index_ = index_ + mrows*mcolumns*int(mosaic_data["layers"])
        section_json["channels"] = list(range(1, int(mosaic_data["channels"]) + 1))
        section_json["tiles"] = tiles
        section_json['slice_fname'] = os.path.split(sectionName)[-1] + "_" + str(depth + 1)
        section_json["image_dimensions"] = image_dimensions
    return section_json


def get_section_data(root_dir: str, n_threads: int, depth: int = 1, sectionNum: int = -1):
    """Retrieve and generate section data from root input directory.

    Args:
        root_dir (str): Input directory
        n_threads (int): How many threads to run the section data generation
        depth (int, optional): _description_. Defaults to 1.
        sectionNum (int, optional): Which specific section number to generate information for. 
                                    If set to -1, generates for all sections. Defaults to -1.

    Returns:
        _type_: Section data
    """
    files = glob.glob(root_dir + 'Mosaic*')
    print(root_dir, files)
    # Look for mosaic file
    if len(files) == 0:
        logging.info("Mosaic File Missing.")
    else:
        mosaic_file = files[0]

    mosaic_data = {}
    with open(mosaic_file) as fp:
        for line in fp:
            k,v = line.rstrip("\n").split(":",1)
            mosaic_data[k]=v

    sectionNames = glob.glob(root_dir + mosaic_data["Sample ID"] + "*")
    section_jsons_list = []
    
    # If a specific section number is provided, generate the section data for that section only
    if sectionNum != -1:
        sectionName = os.path.join(root_dir, "{}-{:04d}".format(mosaic_data["Sample ID"], sectionNum + 1))
        section_jsons = [create_section_json(sectionNum, sectionName, mosaic_data)]
        return mosaic_data, section_jsons
    
    # Otherwise, generate section data for all sections
    for d in range(depth):
        section_jsons = Parallel(n_jobs=n_threads)(delayed(create_section_json)(sno, sectionName, mosaic_data, d) 
                                                   for sno,sectionName in enumerate(sectionNames))
        section_jsons_list.append(section_jsons)
    section_jsons = list(itertools.chain.from_iterable(section_jsons_list))

    return mosaic_data, section_jsons


def stitch_section(data: dict, avg_tiles: list, output_dir: str, H, pX_, pY_, 
                   thresh: int = 15, ch: int = None, 
                   save_undistorted: bool = False, median_thresh=None):
    """Stitches the tiles together to create a complete section.

    Args:
        data (dict): Section data
        avg_tiles (list): List of average tiles for each channel
        output_dir (str): Output directory to save stitched images
        H (_type_): Homography information
        pX_ (_type_): _description_
        pY_ (_type_): _description_
        ch (int, optional): Which channel to stitch for. If None is provided, stitch all channels. Defaults to None.
        save_undistorted (bool, optional): Whether or not to save without distortion correction. Defaults to False.
    """
    
    tiles = generate_tiles(data['tiles'], avg_tiles, H, pX_, pY_, thresh, ch, save_undistorted, median_thresh)
    stitcher = Stitcher(data['image_dimensions'], tiles, data['channels'])
    image, missing = stitcher.run()
    del tiles
    missing_tile_paths = get_missing_tile_paths(missing)

    if ch is None:
        for ch in range(image.shape[2]):
            slice_path = os.path.join(output_dir, "stitched_ch{}".format(ch), data['slice_fname'] + "_{}.tif".format(ch))
            print(slice_path)
            write_output(np.ascontiguousarray(image[:,:,ch]), slice_path)
    else:
        slice_path = os.path.join(output_dir, "stitched_ch{}".format(ch), data['slice_fname'] + "_{}.tif".format(ch))
        print(slice_path)
        write_output(np.ascontiguousarray(image[:,:,ch]), slice_path)
        """
        # Writing temp median mask for preview
        print("Writing mask files for preview...")
        mask_tiles = generate_tiles(data['tiles'], avg_tiles, H, pX_, pY_, thresh, ch, save_undistorted, 
                                    median_thresh=20.0)
        mask_path = os.path.join(output_dir, "stitched_ch{}".format(ch), data['slice_fname'] + "_{}_median_mask.tif".format(ch))
        mask_stitcher = Stitcher(data['image_dimensions'], mask_tiles, data['channels'])
        mask, mask_missing = mask_stitcher.run()
        del mask_tiles
        missing_mask_paths = get_missing_tile_paths(mask_missing)
        write_output(np.ascontiguousarray(mask[:,:,ch]), mask_path)
        """
       

if __name__ == '__main__':
    # Setup import setting
    joblib_backend = None
    if sys.platform == 'win32':
        joblib_backend = 'multiprocessing'

    n_threads = -3

    # Setup globally
    corners1 = np.asarray([[33, 10], [796, 21], [30, 813], [793, 818]])
    corners2 = np.asarray([[20, 20], [776, 20], [20, 794], [776, 794]])
    H, _ = cv2.findHomography(corners1, corners2)
    gridp = create_perfect_grid(42, 43, 4, 18)
    gridp = gridp[20:794, 20:776]

    kx, ky = joblib.load("bezier16x.pkl")

    # Double the size to preserve sampling, need to downsample later
    pX_, pY_ = get_deformation_map(gridp.shape[0], gridp.shape[1], kx, ky)


    # Run main
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--depth', default = 1, type=int)
    parser.add_argument('--sectionNum', default = -1, type=int)
    parser.add_argument('--save_undistorted', default=False, type=bool)
    args = parser.parse_args()
    channel = None
    thresh = 15
    median_thresh = 20.0

    root_dir = os.path.join(args.input_dir, '')
    output_dir = os.path.join(args.output_dir, '')
    depth = args.depth
    sectionNum = args.sectionNum
    save_undistorted = args.save_undistorted
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("Creating stitching JSON for sections")
    mosaic_data, section_jsons = get_section_data(root_dir, n_threads, depth, sectionNum)

    channel_count = int(mosaic_data['channels'])
    print("Creating intermediate directories")
    for ch in range(channel_count):
        ch_dir  = os.path.join(output_dir, "stitched_ch{}".format(ch),"")
        if not os.path.isdir(ch_dir):
            os.mkdir(ch_dir)

    if save_undistorted:
        undistorted_dir = output_dir + "/undistorted"

        if not os.path.isdir(undistorted_dir):
            os.mkdir(undistorted_dir)

        for ch in range(channel_count):
            ch_dir = os.path.join(undistorted_dir, "ch{}".format(ch),"")
            if not os.path.isdir(ch_dir):
                os.mkdir(ch_dir)

    # Generate average tiles if all sections are being stitched
    average_tiles = []
    if sectionNum == -1:
        avg_tiles_dir = os.path.join(output_dir,"avg_tiles")
        print("Generating average tiles")
        generate_avg_tiles(section_jsons, avg_tiles_dir, n_threads, median_thresh)
        for i in range(4):
            average_tiles.append(load_average_tile(os.path.join(avg_tiles_dir,"avg_tile_"+str(i)+".tif")))
    # Otherwise, use placeholder average tiles that do not apply any correction.
    else:
        for i in range(4):
            average_tiles.append(np.ones((832,832)))
    print("Stitching sections...")
    #Parallel(n_jobs=1, backend=joblib_backend)(delayed(stitch_section)(section_json,average_tiles, output_dir) for section_json in tqdm(section_jsons))
    Parallel(n_jobs=n_threads, verbose=13)(delayed(stitch_section)(section_json, average_tiles, output_dir, 
                                                                   H, pX_, pY_, thresh, channel, save_undistorted) for section_json in section_jsons)
