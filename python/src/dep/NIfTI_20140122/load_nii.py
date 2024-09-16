import nibabel as nib
import numpy as np
import os
import tempfile
import gzip
import shutil

def load_nii(filename, img_idx=None, dim5_idx=None, dim6_idx=None, dim7_idx=None,
             old_RGB=0, tolerance=0.1, preferredForm='s'):

    if not filename:
        raise ValueError('Usage: nii = load_nii(filename, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx], [old_RGB], [tolerance], [preferredForm])')

    # Default values for optional parameters
    img_idx = img_idx if img_idx is not None else []
    dim5_idx = dim5_idx if dim5_idx is not None else []
    dim6_idx = dim6_idx if dim6_idx is not None else []
    dim7_idx = dim7_idx if dim7_idx is not None else []
    old_RGB = old_RGB if old_RGB is not None else 0
    tolerance = tolerance if tolerance is not None else 0.1
    preferredForm = preferredForm if preferredForm is not None else 's'

    # Check if the file is gzipped and unpack it if necessary
    if filename.endswith('.gz'):
        if not (filename.endswith('.img.gz') or filename.endswith('.hdr.gz') or filename.endswith('.nii.gz')):
            raise ValueError('Please check filename.')

        tmp_dir = tempfile.mkdtemp()
        with gzip.open(filename, 'rb') as f_in:
            with open(os.path.join(tmp_dir, os.path.basename(filename[:-3])), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        filename = os.path.join(tmp_dir, os.path.basename(filename[:-3]))

    # Load the NIFTI file
    nii = nib.load(filename)
    hdr = nii.header
    img = nii.get_fdata()

    # Handle optional dimensions
    if img_idx:
        img = img[:, :, :, img_idx]
    if dim5_idx:
        img = img[:, :, :, :, dim5_idx]
    if dim6_idx:
        img = img[:, :, :, :, :, dim6_idx]
    if dim7_idx:
        img = img[:, :, :, :, :, :, dim7_idx]

    # Perform some of sform/qform transform if needed
    if preferredForm in ['s', 'S']:
        affine = nii.header.get_sform()
    elif preferredForm in ['q', 'Q']:
        affine = nii.header.get_qform()
    else:
        affine = nii.affine

    # Check for tolerance in the affine transformation
    if tolerance < 1.0:
        if not np.allclose(affine, np.eye(4), atol=tolerance):
            raise ValueError('Affine transformation is not within the tolerance level.')

    # Clean up temporary directory if used
    if filename.endswith('.nii'):
        shutil.rmtree(tmp_dir)

    return {
        'hdr': hdr,
        'filetype': 'NIFTI' if filename.endswith('.nii') else 'ANALYZE',
        'fileprefix': os.path.basename(filename).split('.')[0],
        'machine': 'n/a',  # This information is not directly accessible in nibabel
        'img': img,
        'original': hdr
    }

# Example usage:
# nii = load_nii('example.nii.gz')
