import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import tempfile

def print2array(fig=None, res=1, renderer='-opengl', gs_options=None):
    """
    Exports a figure to an image array.

    Parameters:
    fig -- The figure to be exported. Default: current figure.
    res -- Resolution of the output, as a factor of screen resolution. Default: 1.
    renderer -- Renderer to be used. Default: '-opengl'.
    gs_options -- Optional Ghostscript options.

    Returns:
    A -- MxNx3 uint8 image of the figure.
    bcol -- 1x3 uint8 vector of the background color.
    """
    if fig is None:
        fig = plt.gcf()
    
    fig.canvas.draw()  # Ensure rendering is complete

    # Retrieve the background color
    bcol = fig.get_facecolor()
    res_str = int(plt.rcParams['figure.dpi'] * res)

    # Generate temporary file name
    tmp_nam = tempfile.mktemp(suffix='.tif')
    try:
        fig.savefig(tmp_nam, format='tiff', dpi=res_str)
    except Exception as e:
        print("An error occurred while saving the figure: ", e)
        return None, None

    # Read in the generated bitmap
    A = np.array(Image.open(tmp_nam))
    
    # Delete the temporary bitmap file
    os.remove(tmp_nam)
    
    # Set border pixels to the correct color
    if bcol is None:
        bcol = []
    elif bcol == (1, 1, 1, 1):
        bcol = np.array([255, 255, 255], dtype=np.uint8)
    else:
        mask = (A == 255).all(axis=-1)
        rows, cols = np.where(~mask)
        t, b = rows.min(), rows.max()
        l, r = cols.min(), cols.max()
        border_pixels = np.concatenate([
            A[t:b+1, [l, r], :],
            A[[t, b], l:r+1, :]
        ], axis=0)
        bcol = np.median(border_pixels, axis=(0, 1)).astype(np.uint8)
        A[:, :l] = A[:, r+1:] = A[:t] = A[b+1:] = bcol

    return A, bcol

def font_path():
    # This function mimics the font_path function from the MATLAB code
    # and should set the path for Ghostscript fonts.
    # This is more relevant for systems where Ghostscript is used.
    return os.getenv('GS_FONTPATH', '/usr/share/fonts:/usr/local/share/fonts')
