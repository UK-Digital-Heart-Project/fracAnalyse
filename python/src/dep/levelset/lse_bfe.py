import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

def lse_bfe(u0, Img, b, Ksigma, KONE, nu, timestep, mu, epsilon, iter_lse):
    u = u0
    KB1 = convolve(b, Ksigma, mode='constant', cval=0.0)
    KB2 = convolve(b**2, Ksigma, mode='constant', cval=0.0)
    C = updateC(Img, u, KB1, KB2, epsilon)

    KONE_Img = Img**2 * KONE
    u = updateLSF(Img, u, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse)

    Hu = Heaviside(u, epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    b = updateB(Img, C, M, Ksigma)

    return u, b, C

def updateLSF(Img, u0, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse):
    u = u0
    Hu = Heaviside(u, epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    N_class = M.shape[2]
    e = np.zeros(M.shape)
    
    for kk in range(N_class):
        e[:, :, kk] = KONE_Img - 2 * Img * C[kk] * KB1 + C[kk]**2 * KB2

    for kk in range(iter_lse):
        u = NeumannBoundCond(u)
        K = curvature_central(u)
        DiracU = Dirac(u, epsilon)
        ImageTerm = -DiracU * (e[:, :, 0] - e[:, :, 1])
        penalizeTerm = mu * (4 * laplacian(u) - K)
        lengthTerm = nu * DiracU * K
        u = u + timestep * (lengthTerm + penalizeTerm + ImageTerm)

    return u

def updateB(Img, C, M, Ksigma):
    PC1 = np.zeros_like(Img)
    PC2 = np.zeros_like(Img)
    N_class = M.shape[2]
    for kk in range(N_class):
        PC1 = PC1 + C[kk] * M[:, :, kk]
        PC2 = PC2 + C[kk]**2 * M[:, :, kk]

    KNm1 = convolve(PC1 * Img, Ksigma, mode='constant', cval=0.0)
    KDn1 = convolve(PC2, Ksigma, mode='constant', cval=0.0)

    b = KNm1 / KDn1

    return b

def updateC(Img, u, Kb1, Kb2, epsilon):
    Hu = Heaviside(u, epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2))
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    N_class = M.shape[2]
    C_new = np.zeros(N_class)
    
    for kk in range(N_class):
        Nm2 = Kb1 * Img * M[:, :, kk]
        Dn2 = Kb2 * M[:, :, kk]
        C_new[kk] = np.sum(Nm2) / np.sum(Dn2)

    return C_new

def NeumannBoundCond(f):
    g = f.copy()
    g[[0, -1], :] = g[[2, -3], :]
    g[:, [0, -1]] = g[:, [2, -3]]
    return g

def curvature_central(u):
    ux, uy = np.gradient(u)
    normDu = np.sqrt(ux**2 + uy**2 + 1e-10)
    Nx = ux / normDu
    Ny = uy / normDu
    nxx, _ = np.gradient(Nx)
    _, nyy = np.gradient(Ny)
    k = nxx + nyy
    return k

def Heaviside(x, epsilon):
    return 0.5 * (1 + (2 / np.pi) * np.arctan(x / epsilon))

def Dirac(x, epsilon):
    return (epsilon / np.pi) / (epsilon**2 + x**2)

def laplacian(u):
    return convolve(u, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), mode='constant', cval=0.0)
