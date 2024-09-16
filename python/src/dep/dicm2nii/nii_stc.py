import nibabel as nib
import numpy as np
import os

def nii_stc(in_file, out_file=None, timing=None):
    """
    Perform slice timing correction to the input NIfTI data.

    Parameters:
    - in_file: str or nibabel.Nifti1Image, input NIfTI file path or Nifti1Image object.
    - out_file: str, optional, output NIfTI file path.
    - timing: list or numpy array, optional, slice timing information.
    
    Returns:
    - nibabel.Nifti1Image, corrected NIfTI image.
    """
    to_save = out_file is not None
    
    if isinstance(in_file, str):  # file name
        nii = nib.load(in_file)
    elif isinstance(in_file, nib.Nifti1Image):  # NIfTI image object
        nii = in_file
    else:
        raise ValueError('Input must be NIfTI file name or Nifti1Image object.')

    hdr = nii.header
    img = nii.get_fdata()
    nSL = hdr['dim'][3]
    
    if timing is not None:
        if not isinstance(timing, (list, np.ndarray)) or len(timing) != nSL:
            raise ValueError('Timing should have one number per slice.')
        elif np.min(timing) < -1 or np.max(timing) > 1 or np.max(timing) - np.min(timing) > 1:
            raise ValueError('Timing out of range: must be in unit of TR.')
    else:
        try:
            t = hdr['slice_times']  # Assuming this field exists in the header
        except KeyError:
            TR = hdr['pixdim'][4]
            dur = hdr.get('slice_duration', TR / nSL)
            if dur <= 0 or dur > TR / nSL:
                dur = TR / nSL
            t = 0.5 - np.arange(nSL) * dur / TR
            slice_code = hdr.get('slice_code', 0)
            if slice_code == 2:
                t = t[::-1]
            elif slice_code in [3, 5]:
                t = t[[1::2, ::2]]
            elif slice_code in [4, 6]:
                t = t[::-1][[1::2, ::2]]

    if timing is not None and np.any(np.abs(np.diff(t - timing)) > 0.01):
        print('Warning: Provided timing is inconsistent with header timing.')
        t = timing

    nVol = int(hdr['dim'][4])
    nFFT = 2**int(np.ceil(np.log2(nVol + 40)))
    F = np.fft.fftfreq(nFFT, 1.0 / nFFT)
    F = np.exp(-2j * np.pi * F * np.arange(nSL)[:, None])
    
    img = img.astype(np.float32)
    for i in range(nSL):
        y = img[:, :, i, :]
        pad = np.linspace(y[..., -1], y[..., 0], nFFT - nVol, axis=-1)
        y = np.concatenate([y, pad], axis=-1)
        y = np.fft.fft(y, axis=-1)
        y = y * F[i]
        y = np.fft.ifft(y, axis=-1)
        img[:, :, i, :] = np.real(y[..., :nVol])
    
    corrected_nii = nib.Nifti1Image(img, nii.affine, hdr)
    
    if to_save:
        nib.save(corrected_nii, out_file)
    
    return corrected_nii

# Example usage:
# corrected_nii = nii_stc('input_file.nii', 'output_file.nii')
