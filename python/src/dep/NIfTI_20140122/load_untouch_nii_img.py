import numpy as np

def load_untouch_nii_img(hdr, filetype, fileprefix, machine, img_idx=None, dim5_idx=None, dim6_idx=None, dim7_idx=None, old_RGB=0, slice_idx=None):
    if not all([hdr, filetype, fileprefix, machine]):
        raise ValueError('Usage: [img,hdr] = load_untouch_nii_img(hdr, filetype, fileprefix, machine, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx], [old_RGB], [slice_idx]);')
    
    img_idx = img_idx if img_idx is not None and hdr['dime']['dim'][4] >= 1 else []
    dim5_idx = dim5_idx if dim5_idx is not None and hdr['dime']['dim'][5] >= 1 else []
    dim6_idx = dim6_idx if dim6_idx is not None and hdr['dime']['dim'][6] >= 1 else []
    dim7_idx = dim7_idx if dim7_idx is not None and hdr['dime']['dim'][7] >= 1 else []
    old_RGB = old_RGB if old_RGB is not None else 0
    slice_idx = slice_idx if slice_idx is not None and hdr['dime']['dim'][3] >= 1 else []

    check_indices(img_idx, hdr['dime']['dim'][4], "img_idx")
    check_indices(dim5_idx, hdr['dime']['dim'][5], "dim5_idx")
    check_indices(dim6_idx, hdr['dime']['dim'][6], "dim6_idx")
    check_indices(dim7_idx, hdr['dime']['dim'][7], "dim7_idx")
    check_indices(slice_idx, hdr['dime']['dim'][3], "slice_idx")

    img, hdr = read_image(hdr, filetype, fileprefix, machine, img_idx, dim5_idx, dim6_idx, dim7_idx, old_RGB, slice_idx)
    return img, hdr

def check_indices(indices, max_range, name):
    if indices and not isinstance(indices, (list, np.ndarray)):
        raise ValueError(f'"{name}" should be a numerical array.')
    
    if len(set(indices)) != len(indices):
        raise ValueError(f'Duplicate index in "{name}"')
    
    if indices and (min(indices) < 1 or max(indices) > max_range):
        if max_range == 1:
            raise ValueError(f'"{name}" should be 1.')
        else:
            raise ValueError(f'"{name}" should be an integer within the range of [1 {max_range}].')

def read_image(hdr, filetype, fileprefix, machine, img_idx, dim5_idx, dim6_idx, dim7_idx, old_RGB, slice_idx):
    fn = f"{fileprefix}.img" if filetype in {0, 1} else f"{fileprefix}.nii"
    precision_map = {
        1: ('ubit1', 1), 2: ('uint8', 8), 4: ('int16', 16), 8: ('int32', 32), 16: ('float32', 32),
        32: ('float32', 64), 64: ('float64', 64), 128: ('uint8', 24), 256: ('int8', 8), 511: ('float32', 96),
        512: ('uint16', 16), 768: ('uint32', 32), 1024: ('int64', 64), 1280: ('uint64', 64), 1792: ('float64', 128)
    }
    
    precision, bitpix = precision_map.get(hdr['dime']['datatype'], (None, None))
    if precision is None:
        raise ValueError('This datatype is not supported')

    hdr['dime']['bitpix'] = bitpix
    hdr['dime']['dim'][1:] = [max(1, d) for d in hdr['dime']['dim'][1:]]

    with open(fn, 'rb') as f:
        if filetype in {0, 1}:
            f.seek(0, 0)
        else:
            f.seek(int(hdr['dime']['vox_offset']), 0)

        img = np.fromfile(f, dtype=np.dtype(precision))
        d1, d2, d3, d4, d5, d6, d7 = hdr['dime']['dim'][1:8]
        slice_idx = slice_idx or list(range(1, d3 + 1))
        img_idx = img_idx or list(range(1, d4 + 1))
        dim5_idx = dim5_idx or list(range(1, d5 + 1))
        dim6_idx = dim6_idx or list(range(1, d6 + 1))
        dim7_idx = dim7_idx or list(range(1, d7 + 1))

        if hdr['dime']['datatype'] in {32, 1792}:
            img = img.reshape(-1, 2)
            img = img[:, 0] + 1j * img[:, 1]

        img = img.reshape(d1, d2, d3, d4, d5, d6, d7)
        img = img[:, :, slice_idx, :, :, :, :]
        img = img[:, :, :, img_idx, :, :, :]
        img = img[:, :, :, :, dim5_idx, :, :]
        img = img[:, :, :, :, :, dim6_idx, :]
        img = img[:, :, :, :, :, :, dim7_idx]

        if old_RGB and hdr['dime']['datatype'] == 128 and hdr['dime']['bitpix'] == 24:
            img = img.reshape((d1, d2, d3, 3, len(slice_idx), len(img_idx), len(dim5_idx), len(dim6_idx), len(dim7_idx)))
            img = np.transpose(img, (0, 1, 4, 3, 5, 6, 7, 8))
        elif hdr['dime']['datatype'] == 128 and hdr['dime']['bitpix'] == 24:
            img = img.reshape((3, d1, d2, d3, len(slice_idx), len(img_idx), len(dim5_idx), len(dim6_idx), len(dim7_idx)))
            img = np.transpose(img, (1, 2, 3, 0, 4, 5, 6, 7))
        elif hdr['dime']['datatype'] == 511 and hdr['dime']['bitpix'] == 96:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            img = img.reshape((3, d1, d2, d3, len(slice_idx), len(img_idx), len(dim5_idx), len(dim6_idx), len(dim7_idx)))
            img = np.transpose(img, (1, 2, 3, 0, 4, 5, 6, 7))
        else:
            img = img.reshape((d1, d2, len(slice_idx), len(img_idx), len(dim5_idx), len(dim6_idx), len(dim7_idx)))

        hdr['dime']['dim'][3] = len(slice_idx)
        hdr['dime']['dim'][4] = len(img_idx)
        hdr['dime']['dim'][5] = len(dim5_idx)
        hdr['dime']['dim'][6] = len(dim6_idx)
        hdr['dime']['dim'][7] = len(dim7_idx)

        hdr['dime']['glmax'] = float(img.max())
        hdr['dime']['glmin'] = float(img.min())

    return img, hdr
