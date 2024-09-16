import numpy as np

def crop_borders(A, bcol=None, padding=0, crop_amounts=None):
    if padding is None:
        padding = 0
    if crop_amounts is None:
        crop_amounts = [np.nan] * 4
    crop_amounts = crop_amounts + [np.nan] * (4 - len(crop_amounts))

    h, w, c, n = A.shape
    if bcol is None:  # case of transparent bgcolor
        bcol = A[h//2, 0, :, 0]
    if np.isscalar(bcol):
        bcol = np.full((c,), bcol)

    def col(A):
        return A.flatten()

    # Crop margin from left
    if not np.isfinite(crop_amounts[3]):
        bail = False
        for l in range(w):
            for a in range(c):
                if not np.all(col(A[:, l, a, :]) == bcol[a]):
                    bail = True
                    break
            if bail:
                break
    else:
        l = 1 + abs(int(crop_amounts[3]))

    # Crop margin from right
    if not np.isfinite(crop_amounts[1]):
        bcol = A[h//2, w-1, :, 0]
        bail = False
        for r in range(w-1, l-1, -1):
            for a in range(c):
                if not np.all(col(A[:, r, a, :]) == bcol[a]):
                    bail = True
                    break
            if bail:
                break
    else:
        r = w - abs(int(crop_amounts[1]))

    # Crop margin from top
    if not np.isfinite(crop_amounts[0]):
        bcol = A[0, w//2, :, 0]
        bail = False
        for t in range(h):
            for a in range(c):
                if not np.all(col(A[t, :, a, :]) == bcol[a]):
                    bail = True
                    break
            if bail:
                break
    else:
        t = 1 + abs(int(crop_amounts[0]))

    # Crop margin from bottom
    bcol = A[h-1, w//2, :, 0]
    if not np.isfinite(crop_amounts[2]):
        bail = False
        for b in range(h-1, t-1, -1):
            for a in range(c):
                if not np.all(col(A[b, :, a, :]) == bcol[a]):
                    bail = True
                    break
            if bail:
                break
    else:
        b = h - abs(int(crop_amounts[2]))

    if padding == 0:  # no padding
        if not np.array_equal([t, b, l, r], [0, h-1, 0, w-1]):  # Check if we're actually cropping
            padding = 1  # Leave one boundary pixel to avoid bleeding on resize
    elif abs(padding) < 1:  # pad value is a relative fraction of image size
        padding = int(np.sign(padding) * round(np.mean([b - t, r - l]) * abs(padding)))  # ADJUST PADDING
    else:  # pad value is in units of 1/72" points
        padding = round(padding)  # fix cases of non-integer pad value

    if padding > 0:  # extra padding
        # Create an empty image, containing the background color, that has the cropped image size plus the padded border
        B = np.full(((b - t + 1 + padding * 2), (r - l + 1 + padding * 2), c, n), bcol[:, None, None])
        # vA - coordinates in A that contain the cropped image
        vA = [t, b, l, r]
        # vB - coordinates in B where the cropped version of A will be placed
        vB = [padding, (b - t + 1 + padding), padding, (r - l + 1 + padding)]
        # Place the original image in the empty image
        B[vB[0]:vB[1], vB[2]:vB[3], :, :] = A[vA[0]:vA[1]+1, vA[2]:vA[3]+1, :, :]
        A = B
    else:  # extra cropping
        vA = [t - padding, b + padding, l - padding, r + padding]
        A = A[vA[0]:vA[1]+1, vA[2]:vA[3]+1, :, :]
        vB = [np.nan, np.nan, np.nan, np.nan]

    # For EPS cropping, determine the relative BoundingBox - bb_rel
    bb_rel = [(l - 1) / w, (h - b - 1) / h, (r + 1) / w, (h - t + 1) / h]

    return A, vA, vB, bb_rel

# Example usage
# A = np.random.rand(100, 100, 3, 1)  # Replace with actual image data
# bcol = None  # Replace with actual background color if needed
# padding = 0
# crop_amounts = [np.nan, np.nan, np.nan, np.nan]
# A, vA, vB, bb_rel = crop_borders(A, bcol, padding, crop_amounts)
