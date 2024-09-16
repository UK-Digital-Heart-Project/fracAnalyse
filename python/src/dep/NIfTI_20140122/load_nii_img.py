import numpy as np

def load_nii_img(hdr, filetype, fileprefix, machine, img_idx=None, dim5_idx=None, dim6_idx=None, dim7_idx=None, old_RGB=0):
    if hdr is None or filetype is None or fileprefix is None or machine is None:
        raise ValueError('Usage: load_nii_img(hdr, filetype, fileprefix, machine, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx], [old_RGB])')
    
    if img_idx is None or len(img_idx) == 0 or hdr['dime']['dim'][4] < 1:
        img_idx = []

    if dim5_idx is None or len(dim5_idx) == 0 or hdr['dime']['dim'][5] < 1:
        dim5_idx = []

    if dim6_idx is None or len(dim6_idx) == 0 or hdr['dime']['dim'][6] < 1:
        dim6_idx = []

    if dim7_idx is None or len(dim7_idx) == 0 or hdr['dime']['dim'][7] < 1:
        dim7_idx = []

    if old_RGB is None or len(old_RGB) == 0:
        old_RGB = 0

    # Check img_idx
    if len(img_idx) > 0 and not isinstance(img_idx, np.ndarray):
        raise ValueError('"img_idx" should be a numerical array.')

    if len(np.unique(img_idx)) != len(img_idx):
        raise ValueError('Duplicate image index in "img_idx"')

    if len(img_idx) > 0 and (min(img_idx) < 1 or max(img_idx) > hdr['dime']['dim'][4]):
        max_range = hdr['dime']['dim'][4]

        if max_range == 1:
            raise ValueError('"img_idx" should be 1.')
        else:
            range_str = f"1 {max_range}"
            raise ValueError(f'"img_idx" should be an integer within the range of [{range_str}].')

    # Check dim5_idx
    if len(dim5_idx) > 0 and not isinstance(dim5_idx, np.ndarray):
        raise ValueError('"dim5_idx" should be a numerical array.')

    if len(np.unique(dim5_idx)) != len(dim5_idx):
        raise ValueError('Duplicate index in "dim5_idx"')

    if len(dim5_idx) > 0 and (min(dim5_idx) < 1 or max(dim5_idx) > hdr['dime']['dim'][5]):
        max_range = hdr['dime']['dim'][5]

        if max_range == 1:
            raise ValueError('"dim5_idx" should be 1.')
        else:
            range_str = f"1 {max_range}"
            raise ValueError(f'"dim5_idx" should be an integer within the range of [{range_str}].')

    # Check dim6_idx
    if len(dim6_idx) > 0 and not isinstance(dim6_idx, np.ndarray):
        raise ValueError('"dim6_idx" should be a numerical array.')

    if len(np.unique(dim6_idx)) != len(dim6_idx):
        raise ValueError('Duplicate index in "dim6_idx"')

    if len(dim6_idx) > 0 and (min(dim6_idx) < 1 or max(dim6_idx) > hdr['dime']['dim'][6]):
        max_range = hdr['dime']['dim'][6]

        if max_range == 1:
            raise ValueError('"dim6_idx" should be 1.')
        else:
            range_str = f"1 {max_range}"
            raise ValueError(f'"dim6_idx" should be an integer within the range of [{range_str}].')

    # Check dim7_idx
    if len(dim7_idx) > 0 and not isinstance(dim7_idx, np.ndarray):
        raise ValueError('"dim7_idx" should be a numerical array.')

    if len(np.unique(dim7_idx)) != len(dim7_idx):
        raise ValueError('Duplicate index in "dim7_idx"')

    if len(dim7_idx) > 0 and (min(dim7_idx) < 1 or max(dim7_idx) > hdr['dime']['dim'][7]):
        max_range = hdr['dime']['dim'][7]

        if max_range == 1:
            raise ValueError('"dim7_idx" should be 1.')
        else:
            range_str = f"1 {max_range}"
            raise ValueError(f'"dim7_idx" should be an integer within the range of [{range_str}].')

    img, hdr = read_image(hdr, filetype, fileprefix, machine, img_idx, dim5_idx, dim6_idx, dim7_idx, old_RGB)

    return img, hdr

def read_image(hdr, filetype, fileprefix, machine, img_idx, dim5_idx, dim6_idx, dim7_idx, old_RGB):
    # Placeholder for the actual implementation of read_image function
    img = None
    # Your implementation goes here
    return img, hdr
