import numpy as np
import cv2
from skimage.segmentation import flood_fill
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, dilation, disk
from skimage.feature import canny
import matplotlib.pyplot as plt

from levelSetBatch import level_set_batch

def level_set_thresholding(cropI, binaryMask, sigma=4, epsilon=3):
    print("Level set thresholding")
    if prethresI.shape != binaryMask.shape:
        raise ValueError(f"Shape mismatch: prethresI has shape {prethresI.shape}, but binaryMask has shape {binaryMask.shape}")
    prethresI = flood_fill(cropI, ~binaryMask.astype(bool), 0)
    thresI = level_set_batch(prethresI, sigma, epsilon)
    thresI = np.logical_not(thresI) * binaryMask
    return thresI

def compute_largest_area(thresI, fdMode):
    print("Computing largest area")
    if fdMode == 'LV':
        largestAreaMode = 'JS'
    elif fdMode == 'RV':
        largestAreaMode = 'RV'
    
    if largestAreaMode == 'JS':
        largestAreaI = convex_hull_image(thresI)
        stats = regionprops(largestAreaI.astype(int))[0]
        hortLA = stats.major_axis_length / 2
        vertLA = stats.minor_axis_length / 2
        xLA, yLA = stats.centroid
        ySize, xSize = largestAreaI.shape
        xMask, yMask = np.meshgrid(np.arange(xSize), np.arange(ySize))
        lAMask = ((xMask - xLA) / hortLA) ** 2 + ((yMask - yLA) / hortLA) ** 2 <= 1
        compositeLAI = np.logical_or(largestAreaI, lAMask)
        
    elif largestAreaMode == 'PFT':
        largestAreaI = convex_hull_image(thresI)
        stats = regionprops(largestAreaI.astype(int))[0]
        xLA, yLA = stats.centroid
        hortLA = 1.05 * stats.major_axis_length / 2
        vertLA = 1.05 * stats.minor_axis_length / 2
        theta = np.deg2rad(stats.orientation)
        cc, ss = np.cos(theta), np.sin(theta)
        ySize, xSize = largestAreaI.shape
        xx, yy = np.meshgrid(np.arange(xSize), np.arange(ySize))
        XX = cc * (xx - xLA) - ss * (yy - yLA)
        YY = ss * (xx - xLA) + cc * (yy - yLA)
        equivEllipse = (XX ** 2 / hortLA ** 2 + YY ** 2 / vertLA ** 2 <= 1.0)
        delta = round(0.025 * (hortLA + vertLA))
        largestAreaI = dilation(largestAreaI, disk(delta))
        compositeLAI = np.logical_or(largestAreaI, equivEllipse)
        
    elif largestAreaMode == 'RV':
        countPixels = np.count_nonzero(thresI)
        invertThresI = np.logical_not(thresI) * binaryMask
        countInvertPixels = np.count_nonzero(invertThresI)
        intensityPixels = np.sum(cropI * thresI) / countPixels
        intensityInvertPixels = np.sum(cropI * invertThresI) / countInvertPixels
        
        if intensityInvertPixels > intensityPixels:
            thresI = invertThresI
        largestAreaI = convex_hull_image(thresI)
        delta = 2
        largestAreaI = dilation(largestAreaI, disk(delta))
        compositeLAI = largestAreaI
        
    thresI = thresI * compositeLAI
    return thresI

def edge_detection(thresI):
    print("Edge detection")
    edgeI = canny(thresI, sigma=1)
    return edgeI

def box_counting(edgeI):
    print("Box counting")
    # Placeholder function for box-counting algorithm
    # Compute the fractal dimension here
    fd = 1.23  # Dummy value
    bcFig = None  # Placeholder for the box-counting figure
    return fd, bcFig

def fracDimBatch(cropI, binaryMask, sigma=4, epsilon=3, fdMode='LV'):
    print("Fractal dimension batch")
    thresI = level_set_thresholding(cropI, binaryMask, sigma, epsilon)
    thresI = compute_largest_area(thresI, fdMode)
    edgeI = edge_detection(thresI)
    fd, bcFig = box_counting(edgeI)
    return fd, thresI, bcFig
