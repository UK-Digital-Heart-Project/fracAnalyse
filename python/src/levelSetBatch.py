import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

from dep.levelset.normalize01 import normalize01
from dep.levelset.lse_bfe import lse_bfe

def level_set_batch(I, sigma, epsilon):
    print(f"Level set batch with sigma={sigma} and epsilon={epsilon}")
    Img = I.astype(np.float64)
    A = 255
    Img = A * normalize01(Img)
    nu = 0.001 * A ** 2
    
    iter_outer = 100
    iter_inner = 50
    timestep = 0.1
    mu = 1
    c0 = 1
    
    initialLSF = c0 * np.ones_like(Img)
    xSize, ySize = Img.shape
    initialLSF[int(0.20 * xSize):int(0.80 * xSize), int(0.20 * ySize):int(0.80 * ySize)] = -c0
    u = initialLSF
    
    b = np.ones_like(Img)
    
    K = gaussian_filter(Img, sigma=sigma)
    KI = convolve2d(Img, K, mode='same')
    KONE = convolve2d(np.ones_like(Img), K, mode='same')
    
    for n in range(iter_outer):
        u, b, C = lse_bfe(u, Img, b, K, KONE, nu, timestep, mu, epsilon, iter_inner)
    
    contourI = u > 0
    return contourI
