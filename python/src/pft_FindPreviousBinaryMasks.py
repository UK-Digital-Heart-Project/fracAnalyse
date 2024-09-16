import os
import numpy as np
from skimage import io

def pft_find_previous_binary_masks(top_level_output_folder, leaf_folder):
    nslices = 20
    former_binary_masks_present = [False] * nslices
    binary_mask_stack = []
    return_code = "Folder for search does not exist."

    thres_img_folder = os.path.join(top_level_output_folder, 'ThresImg', leaf_folder)
    if not os.path.isdir(thres_img_folder):
        return former_binary_masks_present, binary_mask_stack, return_code

    for n in range(nslices):
        file_name = f'binaryMaskSlice{n+1}Phase1.png'
        if os.path.isfile(os.path.join(thres_img_folder, file_name)):
            former_binary_masks_present[n] = True

    if not any(former_binary_masks_present):
        return_code = "No former ROI's found."
        return former_binary_masks_present, binary_mask_stack, return_code

    first = next(i for i, x in enumerate(former_binary_masks_present) if x)
    last = len(former_binary_masks_present) - next(i for i, x in enumerate(reversed(former_binary_masks_present)) if x) - 1

    m = last - first + 1
    n = sum(former_binary_masks_present[first:last+1])
    
    if m != n:
        return_code = 'Binary mask stack not contiguous.'
        return former_binary_masks_present, binary_mask_stack, return_code

    binary_mask_stack = np.zeros((nslices, nslices, nslices), dtype=bool)
    for p in range(first, last+1):
        file_name = f'binaryMaskSlice{p+1}Phase1.png'
        path_name = os.path.join(thres_img_folder, file_name)
        bm = io.imread(path_name).astype(bool)
        binary_mask_stack[:, :, p] = bm

    return_code = 'OK'
    return former_binary_masks_present, binary_mask_stack, return_code
