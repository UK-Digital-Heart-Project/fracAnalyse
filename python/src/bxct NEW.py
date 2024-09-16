import numpy as np
from skimage.measure import regionprops
from skimage.util import pad
import matplotlib.pyplot as plt
import csv
import os

def pft_JC_bxct(EdgeImage, Slice, OutputFolder):
    # Padding image to have equal dimensions
    props = regionprops(EdgeImage.astype(int))
    bbox = props[0].bbox
    if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) > 0.25 * EdgeImage.shape[0] * EdgeImage.shape[1]:
        EdgeImage = EdgeImage[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    height, width = EdgeImage.shape

    if height > width:
        padWidth = height - width
        padHeight = 0
        dimen = height
    else:
        padHeight = width - height
        padWidth = 0
        dimen = width

    padI = pad(EdgeImage, ((padHeight // 2, padHeight - padHeight // 2), 
                           (padWidth // 2, padWidth - padWidth // 2)), mode='constant')

    nBoxA, boxSizeA = initGrid(padI, dimen)
    nBoxB, boxSizeB = initGrid(np.flipud(padI), dimen)
    nBoxC, boxSizeC = initGrid(np.fliplr(padI), dimen)
    nBoxD, boxSizeD = initGrid(np.flipud(np.fliplr(padI)), dimen)

    nBox = np.min([nBoxA, nBoxB, nBoxC, nBoxD], axis=0)
    boxSize = boxSizeA

    p = np.polyfit(np.log(boxSize), np.log(nBox), 1)
    FD = -p[0]

    # Output the raw numbers for re-formatting
    FileName = f'Box-Count-Slice-{Slice}.csv'
    PathName = os.path.join(OutputFolder, FileName)

    with open(PathName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Box size', 'Box count', 'Ln(Box size)', 'Ln(Box count)', 'p(1)', 'p(2)'])
        for i in range(len(boxSize)):
            writer.writerow([boxSize[i], nBox[i], np.log(boxSize[i]), np.log(nBox[i]), p[0], p[1]])

    # Add the polynomial fit as well
    FileName = f'Polynomial-Fit-Slice-{Slice}.csv'
    PathName = os.path.join(OutputFolder, FileName)

    xx = np.log([2.0, 0.45 * dimen])
    yy = np.polyval(p, xx)

    with open(PathName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ln(Box size)', 'Ln(Box count)'])
        writer.writerow([xx[0], yy[0]])
        writer.writerow([xx[1], yy[1]])

    # Box Count Plot
    plt.figure()
    plt.loglog(boxSizeA, nBoxA, 's-')
    plt.xlabel('r, box size (pixels)')
    plt.ylabel('n(r), box count')

    x1 = np.linspace(2, 0.45 * dimen)
    y1 = np.exp(np.polyval(p, np.log(x1)))
    plt.loglog(x1, y1)

    plt.legend(['Box Count', 'Regression'])

    FileName = f'Box-Count-Plot-Slice-{Slice}-ED.png'
    plt.savefig(os.path.join(OutputFolder, FileName), dpi=300)
    plt.close()

    return FD

def initGrid(Image, dimen):
    startBoxSize = int(0.45 * dimen)
    curBoxSize = startBoxSize

    nBox = np.zeros(startBoxSize - 1)
    boxSize = np.arange(startBoxSize, 1, -1)

    for sizeCount in range(len(boxSize)):
        curBoxSize = boxSize[sizeCount]
        
        for macroY in range(0, dimen, curBoxSize):
            for macroX in range(0, dimen, curBoxSize):
                boxYinit = macroY
                boxXinit = macroX
                boxYend = min(macroY + curBoxSize, dimen)
                boxXend = min(macroX + curBoxSize, dimen)
                
                boxFound = False
                for curY in range(boxYinit, boxYend):
                    for curX in range(boxXinit, boxXend):
                        if Image[curY, curX]:
                            boxFound = True
                            nBox[sizeCount] += 1
                            break
                    if boxFound:
                        break

    return nBox, boxSize
