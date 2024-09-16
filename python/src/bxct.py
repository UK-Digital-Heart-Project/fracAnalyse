import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.util import pad

def bxct(I):
    # Padding image to have equal dimensions
    props = regionprops(I)
    bbox = props[0].bbox

    if bbox[2] * bbox[3] > 0.25 * I.shape[0] * I.shape[1]:
        I = I[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    height, width = I.shape
    if height > width:
        pad_width = height - width
        pad_height = 0
        dimen = height
    else:
        pad_height = width - height
        pad_width = 0
        dimen = width

    padI = pad(I, ((pad_height % 2, pad_height // 2), (pad_width % 2, pad_width // 2)), mode='constant')

    # Initialize grid and perform box-counting
    nBoxA, boxSizeA = init_grid(padI, dimen)
    nBoxB, boxSizeB = init_grid(np.flip(padI, axis=0), dimen)
    nBoxC, boxSizeC = init_grid(np.flip(padI, axis=1), dimen)
    nBoxD, boxSizeD = init_grid(np.flip(np.flip(padI, axis=0), axis=1), dimen)

    nBox = np.minimum.reduce([nBoxA, nBoxB, nBoxC, nBoxD])
    boxSize = boxSizeA

    # Linear regression to find fractal dimension
    p = np.polyfit(np.log(boxSize), np.log(nBox), 1)
    fd = -p[0]

    # Box Count Plot
    plt.figure()
    plt.loglog(boxSizeA, nBoxA, 's-')
    x1 = np.linspace(2, 0.45 * dimen, num=100)
    y1 = np.exp(np.polyval(p, np.log(x1)))
    plt.loglog(x1, y1)
    plt.xlabel('r, box size (pixels)')
    plt.ylabel('n(r), box count')
    plt.legend(['Box Count', 'Regression'])
    plt.show()

    return fd

def init_grid(I, dimen):
    start_box_size = int(0.45 * dimen)
    nBox = np.zeros(start_box_size - 1)
    boxSize = np.arange(start_box_size, 1, -1)
    
    for sizeCount, curBoxSize in enumerate(boxSize):
        for macroY in range(int(np.ceil(dimen / curBoxSize))):
            for macroX in range(int(np.ceil(dimen / curBoxSize))):
                boxYinit = macroY * curBoxSize
                boxXinit = macroX * curBoxSize
                boxYend = min(boxYinit + curBoxSize, dimen)
                boxXend = min(boxXinit + curBoxSize, dimen)

                boxFound = np.any(I[boxYinit:boxYend, boxXinit:boxXend])
                if boxFound:
                    nBox[sizeCount] += 1

    return nBox, boxSize
