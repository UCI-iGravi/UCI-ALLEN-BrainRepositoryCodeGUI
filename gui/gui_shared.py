import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage


from cellAnalysis.cell_detection import createShardedPointAnnotation, readSectionTif


def auto_contrast(data: np.ndarray, alpha: float = None, beta: float = None) -> np.ndarray:
    """
    Preprocess tiff files to automatically adjust brightness and contrast.
    https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    """
    if not alpha:
        alpha = np.iinfo(data.dtype).max / (np.max(data) - np.min(data))
    if not beta:
        beta = -np.min(data) * alpha
    img = cv2.convertScaleAbs(data.copy(), alpha=alpha, beta=beta)
    return img


def preprocess(img: np.ndarray, min_val: float = None, max_val: float = 400) -> np.ndarray:
    """
    Preprocesses volume data. Clips maximum value at max_val and then normalizes volume
    between 0-255.
    """
    data = img.copy()
    data[data > max_val] = max_val
    if min_val:
        data[data < min_val] = min_val
    data = cv2.normalize(src=data, dst=None, alpha=0, beta=255, 
                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return data


def remove_artifacts(img: np.ndarray, thresh=None,
                     min_size=None, area_threshold=None, num_iter=None, erode=True, convex_hull=True, 
                     debug=False) -> np.ndarray:
    mask = img.copy()
    y, x = img.shape
    
    # Set parameters
    if min_size is None:
        min_size = max(64, int(max(y, x) * 0.20))      # Largest allowable size for specks
    if area_threshold is None:
        area_threshold = max(64, int(max(y, x) * 20))  # Fill in holes smaller than this size
    if num_iter is None:
        num_iter = int(max(y, x) / 500)                # 20, original dilation factor 500
    
    if debug:
        plt.imshow(mask)
        plt.title('Original')
        plt.show()

    # Preprocess curr_img
    mask = preprocess(mask, min_val=0, max_val=400)
    
    if debug:
        plt.imshow(mask)
        plt.title('Preprocessed')
        plt.show()
    
    # Binarize / threshold
    if not thresh:  # Adaptive threshold if no threshold is given
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype(bool)
    else:  # Otherwise, use given threshold
        mask = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)[1].astype(bool)
    mask = np.where(mask > 0, 1, 0).astype(bool)
    
    if debug:
        plt.imshow(mask)
        plt.title('Binarized')
        plt.show()
    
    # Remove small specks from background
    mask = skimage.morphology.remove_small_objects(mask.astype(bool), min_size=min_size).astype(bool)
    
    if debug:
        plt.imshow(mask)
        plt.title('Removed small specks')
        plt.show()
    
    # Expand morphology around brain mask to capture surrounding areas 
    # that may have been potentially filtered out.
    start = time.time()
    mask = binary_dilation(mask, iterations=num_iter).astype(bool)
    elapsed = time.time() - start
    
    if debug:
        print('Binary dilation elapsed time:', elapsed)
        plt.imshow(mask)
        plt.title('Dilated')
        plt.show()
    
    # Fill in holes
    mask = skimage.morphology.remove_small_holes(mask, area_threshold=area_threshold).astype(bool)
    
    if debug:
        plt.imshow(mask)
        plt.title('Filled in holes')
        plt.show()
        
    # Erode borders to get rid of artifacting if needed
    if erode:
        start = time.time()
        mask = binary_erosion(mask, iterations=num_iter * 4).astype(bool)
        elapsed = time.time() - start
        
        if debug:
            print('Binary erosion elapsed time:', elapsed)
            plt.imshow(mask)
            plt.title('Erosion')
            plt.show()
        
    # Use convex hull if checked
    if convex_hull:
        mask = skimage.morphology.convex_hull_image(mask)

    # Apply mask to original image
    masked_img = np.where(mask, img, 0)
    
    if debug:
        masked_img_preprocessed = preprocess(masked_img, min_val=0, max_val=400)
        plt.imshow(masked_img_preprocessed)
        plt.title('Mask applied to original (preprocessed)')
        plt.show()
    
    return masked_img


def get_cell_locations(img, index = None,
                       intensity_threshold = 5, min_distance = 5, size_threshold = 15,
                       bg_threshold = 15, min_size = 64, area_threshold = 64, dilation_iter = 20):
    """
    img : 2d Image array or img_path
    intensity_threshold: backgound intensity in 0 to 255 scale
    size_threshold : Minimum cell size
    min_distance: Minimum distance between two cell centers
    
    index : Cell z index
    """
    
    if type(img) == str:
        img = readSectionTif(img)
        
    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Show the original section
    #axs[0].set_title("Original section " + str(index))
    #axs[0].imshow(auto_contrast(img, alpha=0.25))
        
    # Normalize the section to highlight larger values
    section = (img/np.iinfo(img.dtype).max)*255
    section = section.astype(np.uint8)
    #axs[1].set_title("Normalized section " + str(index))
    #axs[1].imshow(auto_contrast(section, alpha=40))
    #plt.show()
    
    
    ##### MY BACKGROUND REMOVAL #####
    y, x = section.shape

    
    #fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Remove the background 1 (old method)
    #bg = remove_background(section)
    #axs[0].set_title("Background removal (old method)")
    #axs[0].imshow(auto_contrast(bg, alpha=40))
    
    # Remove the background 2 (my method)
    #THRESH = 15  # Background thresh
    #MIN_SIZE = max(64, int(max(y, x) * 0.20))      # Largest allowable size for specks
    #AREA_THRESHOLD = max(64, int(max(y, x) * 20))  # Fill in holes smaller than this size
    #DILATION_ITER = int(max(y, x) / 500)           # 20
    bg_mask = remove_artifacts(img, thresh=bg_threshold, min_size=min_size, area_threshold=area_threshold, num_iter=dilation_iter, 
                               erode=True, convex_hull=False, debug=False)
    bg_mask[bg_mask != 0] = 1  # Convert to binary
    #axs[2].set_title('New background removal (mask)')
    #axs[2].imshow(auto_contrast(bg_mask, alpha=40))
    # Apply mask to normalized section
    bg_new = section.copy()
    bg_new[bg_mask == 0] = 0
    #axs[1].set_title("New background removal (overlay)")
    #axs[1].imshow(auto_contrast(bg_new, alpha=40))
    #axs[1].contour(bg_mask, colors='r', levels=[0.5])
    #plt.show()
    
    ##### DO PEAK DETECTION #####
    #fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Local peak detection 1 (no BG removal)
    #peaks = peak_local_max(section, min_distance=min_distance, threshold_abs=intensity_threshold)
    #axs[0].set_title("Local peak detection (No BG removal)")
    #axs[0].imshow(auto_contrast(img, alpha=0.25))
    #axs[0].plot(peaks[:, 1], peaks[:, 0], 'r.')
    
    # Local peak detection 2
    #peaks = peak_local_max(bg, min_distance=min_distance, threshold_abs=intensity_threshold)
    #axs[1].set_title("Local peak detection (old BG removal)")
    #axs[1].imshow(auto_contrast(img, alpha=0.25))
    #axs[1].plot(peaks[:, 1], peaks[:, 0], 'r.')
    
    # Local peak detection 3
    peaks = peak_local_max(bg_new, min_distance=min_distance, threshold_abs=intensity_threshold)
    #axs[2].set_title("Local peak detection (new BG removal)")
    #axs[2].imshow(auto_contrast(img, alpha=0.25))
    #axs[2].plot(peaks[:, 1], peaks[:, 0], 'r.')
    #plt.show()
    
    # Zoom in original
    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #axs[0].set_title("Zoomed-in original section " + str(index))
    #axs[0].imshow(auto_contrast(img, alpha=0.05))
    #axs[0].set_xlim([6500, 8500])
    #axs[0].set_ylim([3200, 2400])
    
    # Zoom in peak detection
    #axs[1].set_title("Zoomed-in detected local peaks (new BG removal)")
    #axs[1].imshow(auto_contrast(img, alpha=0.05))
    #axs[1].plot(peaks[:, 1], peaks[:, 0], 'r.')
    #axs[1].set_xlim([6500, 8500])
    #axs[1].set_ylim([3200, 2400])
    #plt.show()

    bg = bg_new

    # Create a mask from the local peak coordinates for watershed and use for size detection
    mask = np.zeros(bg.shape)
    mask[tuple(peaks.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    #print("Size Detection")
    watershed_mask = bg > intensity_threshold
    labels = watershed(-bg, markers, mask = watershed_mask)

    # Compute the center of mass of each cell
    centers = np.array(scipy.ndimage.center_of_mass(bg, labels, index=np.arange(1, np.max(labels) + 1)))
    centers = centers.astype(int)

    # Filter out small cells
    sizes = scipy.ndimage.sum(np.ones(labels.shape, dtype=bool), labels=labels, index=np.arange(1, np.max(labels) + 1))
    print("Centers before filtering:", centers.shape)
    print("centers:", centers)
    cells = centers[sizes >= size_threshold]
    print("cells:", cells)
    
    if index is not None:
        print("get_cell_locations")
        print("\tindex:", index)
        print("\tbefore cells.shape:", cells.shape)
        print("\tconstructed shape:", np.zeros((cells.shape[0], 1)).shape)
        if cells.size:
            cells = np.hstack([np.zeros((cells.shape[0], 1)) + index, cells])
        else:
            cells = np.array([])
        print("\tafter cells.shape:", cells.shape)
        print("\tcells:")
        print(cells)
        print()
        
    # Show final result
    #plt.title("Final result")
    #plt.imshow(auto_contrast(img, alpha=0.25))
    #plt.plot(cells[:, 2], cells[:, 1], 'r.')
    #plt.show()
    
    # Create a new figure with 2 subplots side by side
    #fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot on the first subplot
    #axs[0].imshow(auto_contrast(img, alpha=0.05))
    #axs[0].set_title('Original zoomed-in section ' + str(index))
    #axs[0].set_xlim([6500, 8500])
    #axs[0].set_ylim([3200, 2400])

    # Plot on the second subplot
    #axs[1].imshow(auto_contrast(img, alpha=0.05))
    #axs[1].plot(cells[:, 2], cells[:, 1], 'r.')
    #axs[1].set_title('Final result - zoomed-in counted cells')
    #axs[1].set_xlim([6500, 8500])
    #axs[1].set_ylim([3200, 2400])

    # Display the figure with the subplots
    #plt.show()
 
    print("Number of detected cells:", len(cells), "\n")
    #print("Detected cells:")
    #pprint(cells)
    
    return cells