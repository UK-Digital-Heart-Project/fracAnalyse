import os
import shutil
import gzip
import multiprocessing
import numpy as np
import pydicom
import json
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from nibabel import Nifti1Image, save
from tkinter import filedialog, Tk
import requests
from datetime import datetime

from dicm_hdr import dicm_hdr
from dicm_dict import dicm_dict
from dicm_img import dicm_img

def dicm2nii(src, data_folder, fmt='nii.gz'):
    # Initial setup
    ext = '.nii'
    if isinstance(fmt, str):
        if 'hdr' in fmt.lower() or 'img' in fmt.lower():
            ext = '.img'
        if 'gz' in fmt.lower():
            ext += '.gz'
        if '3D' in fmt.lower():
            rst3D = True
    elif isinstance(fmt, int):
        if fmt in [0, 1, 4, 5]:
            ext = '.nii'
        elif fmt in [2, 3, 6, 7]:
            ext = '.img'
        if fmt % 2 != 0:
            ext += '.gz'
        rst3D = fmt > 3
    else:
        raise ValueError('Invalid output file format (the 3rd input).')

    # Handle source
    if isinstance(src, str):
        if os.path.isdir(src):
            dcm_folder = src
        elif os.path.isfile(src):
            dcm_folder = os.path.dirname(src)
            if src.endswith('.zip') or src.endswith('.tgz'):
                dcm_folder = os.path.join(data_folder, 'tmpDcm')
                shutil.unpack_archive(src, dcm_folder)
        else:
            raise ValueError('Unknown dicom source.')
    elif isinstance(src, list):
        dcm_folder = os.path.dirname(src[0])
    else:
        raise ValueError('Invalid dicom source.')

    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Collect all DICOM files
    fnames = []
    for root, _, files in os.walk(dcm_folder):
        for file in files:
            if file.endswith('.dcm'):
                fnames.append(os.path.join(root, file))

    if not fnames:
        raise ValueError('No DICOM files found.')

    # Process each DICOM file and convert to NIfTI
    for fname in fnames:
        try:
            ds = pydicom.dcmread(fname)
            if hasattr(ds, 'PixelData'):
                img_data = ds.pixel_array
                nifti_img = Nifti1Image(img_data, np.eye(4))  # Identity affine
                nifti_filename = os.path.join(data_folder, f"{os.path.basename(fname).split('.')[0]}{ext}")
                save(nifti_img, nifti_filename)
        except InvalidDicomError as e:
            print(f"Skipping non-DICOM file: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("Conversion complete.")

# Function: return folder name for a file name
def folder_from_file(fname):
    folder = os.path.dirname(fname)
    if not folder:
        folder = os.getcwd()
    return folder

# Function: return PatientName
def patient_name(s):
    subj = try_get_field(s, 'PatientName')
    if not subj:
        subj = try_get_field(s, 'PatientID', 'Anonymous')
    return subj

# Function: return SeriesDescription
def protocol_name(s):
    name = try_get_field(s, 'SeriesDescription')
    if not name or (s.get('Manufacturer', '').startswith('SIEMENS') and
                    len(name) > 9 and name.endswith('MoCoSeries')):
        name = try_get_field(s, 'ProtocolName')
    if not name:
        name = os.path.splitext(s.get('Filename', ''))[0]
    return name

# Function: return true if keyword is in s.ImageType
def is_type(s, keyword):
    typ = try_get_field(s, 'ImageType', '')
    return keyword in typ

# Function: return true if series is DTI
def is_dti(s):
    if is_type(s, '\\DIFFUSION'):
        return True
    if s.get('Manufacturer', '').startswith('GE'):
        return try_get_field(s, 'DiffusionDirection', 0) > 0
    elif s.get('Manufacturer', '').lower().startswith('philips'):
        return try_get_field(s, 'MRSeriesDiffusion', 'N') == 'Y'
    else:
        return csa_header(s, 'B_value') is not None

# Function: get field if exists, return default value otherwise
def try_get_field(s, field, dft_val=None):
    return s.get(field, dft_val)

# Function: Set most nii header and re-orient img
def set_nii_header(nii, s, save_pn):
    dim = nii['hdr']['dim'][1:4]
    ixyz, R, pixdim, xyz_unit = xform_mat(s, dim)
    R[0:2, :] = -R[0:2, :]  # dicom LPS to nifti RAS, xform matrix before reorient

    ph_pos, i_phase = phase_direction(s)
    fps_bits = [0, 0, 16]
    if i_phase == 2:
        fps_bits = [1, 4, 16]
    elif i_phase == 1:
        fps_bits = [4, 1, 16]

    s, nii['hdr'] = slice_timing(s, nii['hdr'])
    nii['hdr']['xyzt_units'] = xyz_unit + nii['hdr']['xyzt_units']

    _, perm = zip(*sorted(enumerate(ixyz), key=lambda x: x[1]))
    if (try_get_field(s, 'MRAcquisitionType', '') == '3D' or s.get('isDTI')) and \
            dim[2] > 1 and perm != tuple(range(3)):
        R[:, :3] = R[:, perm]
        fps_bits = [fps_bits[i] for i in perm]
        ixyz = [ixyz[i] for i in perm]
        dim = [dim[i] for i in perm]
        pixdim = [pixdim[i] for i in perm]
        nii['hdr']['dim'][1:4] = dim
        nii['img'] = np.transpose(nii['img'], perm + (3, 4, 5, 6, 7))
        if 'bvec' in s:
            s['bvec'] = s['bvec'][:, perm]

    i_sl = fps_bits.index(16)
    i_phase = fps_bits.index(4)

    nii['hdr']['dim_info'] = sum((i + 1) * fps_bits[i] for i in range(3))
    nii['hdr']['pixdim'][1:4] = pixdim

    ind4 = [ixyz[i] + 4 * i for i in range(3)]
    flp = [R[ind4[i], i] < 0 for i in range(3)]
    d = np.linalg.det(R[:3, :3]) * np.prod([1 - 2 * f for f in flp])
    if d > 0:
        flp[0] = not flp[0]
    rot_m = np.diag([1 - 2 * int(f) for f in flp] + [1])
    rot_m[:3, 3] = [(dim[i] - 1) * flp[i] for i in range(3)]
    R = np.linalg.inv(rot_m).dot(R)
    for k in range(3):
        if flp[k]:
            nii['img'] = np.flip(nii['img'], k)
    if flp[i_phase]:
        ph_pos = not ph_pos
    if 'bvec' in s:
        s['bvec'][:, flp] = -s['bvec'][:, flp]
    if flp[i_sl] and 'SliceTiming' in s:
        s['SliceTiming'] = s['SliceTiming'][::-1]
        sc = nii['hdr']['slice_code']
        if sc > 0:
            nii['hdr']['slice_code'] = sc + (sc % 2) * 2 - 1

    frm_code = all(k in s for k in ['ImageOrientationPatient', 'ImagePositionPatient'])
    frm_code = try_get_field(s, 'TemplateSpace', frm_code)
    nii['hdr']['sform_code'] = frm_code
    nii['hdr']['srow_x'] = R[0, :]
    nii['hdr']['srow_y'] = R[1, :]
    nii['hdr']['srow_z'] = R[2, :]

    if abs(np.sum(R[:, i_sl] ** 2) - pixdim[i_sl] ** 2) < 0.01:
        nii['hdr']['qform_code'] = frm_code
        nii['hdr']['qoffset_x'] = R[0, 3]
        nii['hdr']['qoffset_y'] = R[1, 3]
        nii['hdr']['qoffset_z'] = R[2, 3]

        R = R[:3, :3]
        R = R / np.sqrt(np.sum(R ** 2, axis=0))
        q, nii['hdr']['pixdim'][0] = dcm2quat(R)
        nii['hdr']['quatern_b'] = q[1]
        nii['hdr']['quatern_c'] = q[2]
        nii['hdr']['quatern_d'] = q[3]

    str = try_get_field(s, 'ImageComments', '')
    if is_type(s, '\\MOCO\\'):
        str = ''
    foo = try_get_field(s, 'StudyComments')
    if foo:
        str = f"{str};{foo}"
    str = f"{str};{s['Manufacturer'].split()[0]}"
    foo = try_get_field(s, 'ProtocolName')
    if foo:
        str = f"{str};{foo}"
    nii['hdr']['aux_file'] = str

    seq = asc_header(s, 'tSequenceFileName')
    if not seq:
        seq = try_get_field(s, 'ScanningSequence')
    else:
        ind = seq.rfind('\\')
        if ind != -1:
            seq = seq[ind + 1:]
    if save_pn:
        nii['hdr']['db_name'] = patient_name(s)
    nii['hdr']['intent_name'] = seq

    if 'AcquisitionDateTime' not in s and 'AcquisitionTime' in s:
        s['AcquisitionDateTime'] = f"{try_get_field(s, 'AcquisitionDate', '')}{try_get_field(s, 'AcquisitionTime', '')}"
    foo = try_get_field(s, 'AcquisitionDateTime')
    descrip = f"time={foo[:18]};"
    TE0 = asc_header(s, 'alTE[0]') / 1000
    TE1 = asc_header(s, 'alTE[1]') / 1000
    if TE1:
        s['SecondEchoTime'] = TE1
    dTE = abs(TE1 - TE0) if TE1 else None
    if not TE0:
        TE0 = try_get_field(s, 'EchoTime')
    if not dTE and try_get_field(s, 'NumberOfEchoes', 1) > 1:
        dTE = try_get_field(s, 'SecondEchoTime') - TE0
    if dTE:
        descrip = f"dTE={dTE};{descrip}"
        s['deltaTE'] = dTE
    elif TE0:
        descrip = f"TE={TE0};{descrip}"

    if try_get_field(s, 'MRAcquisitionType') != '3D' and i_phase is not None:
        hz = csa_header(s, 'BandwidthPerPixelPhaseEncode')
        dwell = 1000 / hz / dim[i_phase] if hz else None
        if not dwell:
            lns = asc_header(s, 'sKSpace.lPhaseEncodingLines')
            dur = csa_header(s, 'SliceMeasurementDuration')
            dwell = dur / lns if dur else None
        if not dwell:
            dur = csa_header(s, 'RealDwellTime') * 1e-6
            dwell = dur * asc_header(s, 'sKSpace.lBaseResolution') if dur else None
        if not dwell:
            dwell = try_get_field(s, 'EffectiveEchoSpacing') / 1000
        if not dwell:
            wfs = try_get_field(s, 'WaterFatShift')
            epi_factor = try_get_field(s, 'EPIFactor')
            dwell = wfs / (434.215 * (epi_factor + 1)) * 1000 if wfs else None
        if dwell:
            s['EffectiveEPIEchoSpacing'] = dwell
            epi_factor = try_get_field(s, 'EPIFactor')
            if not epi_factor:
                epi_factor = asc_header(s, 'sFastImaging.lEPIFactor')
            pat = try_get_field(s, 'ParallelReductionFactorInPlane', 1)
            if not pat:
                pat = asc_header(s, 'sPat.ucPATMode')
            readout = dwell * pat * epi_factor / 1000 if epi_factor else None
            if readout:
                s['ReadoutSeconds'] = readout
            descrip = f"readout={readout};{descrip}" if s.get('isDTI') and readout else f"dwell={dwell};{descrip}"

    if i_phase is not None:
        if ph_pos is None:
            pm, b67 = '?', 0
        elif ph_pos:
            pm, b67 = '', 1
        else:
            pm, b67 = '-', 2
        nii['hdr']['dim_info'] += b67 * 64
        axes = 'xyz'
        ph_dir = f"{pm}{axes[i_phase]}"
        s['UnwarpDirection'] = ph_dir
        descrip = f"phase={ph_dir};{descrip}"
    nii['hdr']['descrip'] = descrip

    if any(k in s for k in ['RescaleSlope', 'RescaleIntercept']):
        slope = try_get_field(s, 'RescaleSlope', 1)
        inter = try_get_field(s, 'RescaleIntercept', 0)
        val = sorted([(max(nii['img'].ravel()) * slope + inter), (min(nii['img'].ravel()) * slope + inter)])
        d_class = nii['img'].dtype
        if np.issubdtype(d_class, np.float) or (slope.is_integer() and inter.is_integer() and
                                                 val[0] >= np.iinfo(d_class).min and val[1] <= np.iinfo(d_class).max):
            nii['img'] = nii['img'] * slope + inter
        else:
            nii['hdr']['scl_slope'] = slope
            nii['hdr']['scl_inter'] = inter

# Function: reshape mosaic into volume, remove padded zeros
def mos2vol(mos, nSL):
    nMos = int(np.ceil(np.sqrt(nSL)))  # always nMos x nMos tiles
    nr, nc, nv = mos.shape  # number of row, col, and vol in mosaic

    nr = nr // nMos
    nc = nc // nMos
    vol = np.zeros((nr, nc, nSL, nv), dtype=mos.dtype)
    for i in range(nSL):
        r = (i % nMos) * nr
        c = (i // nMos) * nc
        vol[:, :, i, :] = mos[r:r+nr, c:c+nc, :]
    return vol

# Subfunction: set slice timing related info
def slice_timing(s, hdr):
    TR = try_get_field(s, 'RepetitionTime')  # in ms
    if TR is None:
        TR = try_get_field(s, 'TemporalResolution')
    if TR is None:
        return s, hdr
    hdr['pixdim'][4] = TR / 1000
    if try_get_field(s, 'isDTI', 0):
        return s, hdr
    hdr['xyzt_units'] = 8  # seconds
    if hdr['dim'][4] < 3:
        return s, hdr  # skip structural, fieldmap etc.

    delay = asc_header(s, 'lDelayTimeInTR') / 1000  # in ms now
    if delay is None:
        delay = 0
    TA = TR - delay
    t = csa_header(s, 'MosaicRefAcqTimes')  # in ms
    if t is not None and 'LastFile' in s and max(t) - min(t) > TA:
        try:
            t = mb_slicetiming(s, TA)
        except:
            pass
    if t is None:
        t = try_get_field(s, 'RefAcqTimes')  # GE or Siemens non-mosaic

    nSL = hdr['dim'][3]
    if t is None:  # non-mosaic Siemens: create 't' based on ucMode
        ucMode = asc_header(s, 'sSliceArray.ucMode')  # 1/2/4: Asc/Desc/Inter
        if ucMode is None:
            return s, hdr
        t = np.arange(nSL) * TA / nSL
        if ucMode == 2:
            t = t[::-1]
        elif ucMode == 4:
            if nSL % 2:
                t = np.concatenate((t[::2], t[1::2]))
            else:
                t = np.concatenate((t[1::2], t[::2]))
        if asc_header(s, 'sSliceArray.ucImageNumb'):
            t = t[::-1]
        s['RefAcqTimes'] = t

    if len(t) < 2:
        return s, hdr
    t = t - min(t)  # it may be relative to the first slice

    t1 = np.sort(t)
    dur = np.sum(np.diff(t1)) / (nSL - 1)
    dif = np.sum(np.diff(t)) / (nSL - 1)
    if dur == 0 or (t1[-1] > TA):
        sc = 0  # no useful info, or bad timing MB
    elif t1[0] == t1[1]:
        sc = 7
        t1 = np.unique(t1)  # made-up code 7 for MB
    elif abs(dif - dur) < 1e-3:
        sc = 1  # ascending
    elif abs(dif + dur) < 1e-3:
        sc = 2  # descending
    elif t[0] < t[2]:  # ascending interleaved
        if t[0] < t[1]:
            sc = 3  # odd slices first
        else:
            sc = 5  # Siemens even number of slices
    elif t[0] > t[2]:  # descending interleaved
        if t[0] > t[1]:
            sc = 4
        else:
            sc = 6  # Siemens even number of slices
    else:
        sc = 0  # unlikely to reach

    s['SliceTiming'] = 0.5 - t / TR  # as for FSL custom timing
    hdr['slice_code'] = sc
    hdr['slice_end'] = nSL - 1  # 0-based, slice_start default to 0
    hdr['slice_duration'] = min(np.diff(t1)) / 1000
    return s, hdr

def get_dti_para(h, nii):
    nSL = nii.header['dim'][3]
    nDir = nii.header['dim'][4]
    if nDir < 2:
        return h, nii
    
    bval = np.full((nDir,), np.nan)
    bvec = np.full((nDir, 3), np.nan)
    s = h[0]
    
    if hasattr(s, 'bvec_original'):
        bval = s.B_value
        bvec = s.bvec_original
    elif hasattr(s, 'PerFrameFunctionalGroupsSequence'):
        if try_get_field(s, 'Dim3IsVolume', False):
            iDir = np.arange(1, nDir+1)
        else:
            iDir = np.arange(1, nSL * nDir, nSL)
        
        s2 = {'B_value': np.nan * np.ones(nDir), 'DiffusionGradientDirection': np.nan * np.ones((3, nDir))}
        s2 = dicm_hdr(s, s2, iDir)  # Calls a hypothetical `dicm_hdr` function for DICOM header parsing
        bval = s2['B_value']
        bvec = s2['DiffusionGradientDirection'].T
    else:
        dict_keys = ['B_value', 'B_factor', 'SlopInt_6_9', 'DiffusionDirectionX', 'DiffusionDirectionY', 'DiffusionDirectionZ']
        iDir = (np.arange(0, nDir) * len(h) / nDir).astype(int) + 1
        
        for j in range(nDir):
            s2 = h[iDir[j]]
            val = try_get_field(s2, 'B_value')
            if val == 0:
                continue
            vec = try_get_field(s2, 'DiffusionGradientDirection')
            imgRef = vec is None  # If no vector, likely GE
            
            if val is None or vec is None:
                s2 = dicm_hdr(s2.Filename, dict_keys)
            
            if val is None:
                val = try_get_field(s2, 'B_factor')
            if val is None and 'SlopInt_6_9' in s2:
                val = s2['SlopInt_6_9'][0]
            if val is None:
                val = 0
            
            bval[j] = val
            
            if vec is None:
                vec = np.array([
                    try_get_field(s2, 'DiffusionDirectionX', 0),
                    try_get_field(s2, 'DiffusionDirectionY', 0),
                    try_get_field(s2, 'DiffusionDirectionZ', 0)
                ])
            
            bvec[j, :] = vec
    
    if np.isnan(bval).all() and np.isnan(bvec).all():
        raise ValueError(f"Failed to get DTI parameters: {s['NiftiName']}")
    
    bval[np.isnan(bval)] = 0
    bvec[np.isnan(bvec)] = 0
    
    if 'Philips' in s.Manufacturer[:7]:
        ind = np.where((bval > 1e-4) & (np.sum(np.abs(bvec), axis=1) < 1e-4))[0]
        if ind.size > 0:
            try:
                isISO = s.LastFile['DiffusionDirectionality']
            except KeyError:
                isISO = False
            if not isISO:
                bval = np.delete(bval, ind)
                bvec = np.delete(bvec, ind, axis=0)
                nii.img = np.delete(nii.img, ind, axis=-1)
                nDir -= len(ind)
                nii.header['dim'][4] = nDir
    
    h[0]['bval'] = bval
    h[0]['bvec_original'] = bvec
    
    ixyz, R = xform_mat(s, nii.header['dim'][1:4])  # Transform matrix
    if 'imgRef' in locals() and imgRef:
        if try_get_field(s, 'InPlanePhaseEncodingDirection') == 'ROW':
            bvec = bvec[:, [1, 0, 2]]
            bvec[:, 1] = -bvec[:, 1]
            if ixyz[2] < 3:
                raise ValueError(f"bvec sign issue in non-axial acquisition for {s['NiftiName']}")
        
        flp = R[ixyz + np.array([0, 4, 8])] < 0
        flp[2] = not flp[2]  # GE slice direction is opposite
        if ixyz[2] == 1:
            flp[0] = not flp[0]
        
        for i in range(3):
            if flp[i]:
                bvec[:, i] = -bvec[:, i]
    else:
        R = R[:3, :3]
        R = R / np.sqrt(np.sum(R**2, axis=0))
        bvec = np.dot(bvec, R)
    
    h[0]['bvec'] = bvec
    return h, nii

def save_dti_para(s, fname):
    # Check if 'bvec' exists and is not all zeros
    if not hasattr(s, 'bvec') or np.all(s.bvec == 0):
        return
    
    # If 'bval' exists, save it to a .bval file
    if hasattr(s, 'bval'):
        with open(f'{fname}.bval', 'w') as f:
            f.write('\t'.join(map(str, s.bval)) + '\n')  # One row of bvals

    # Prepare format string for bvec, 3 rows and 'n' columns
    format_str = ''.join(['%9.6f\t'] * s.bvec.shape[1]) + '\n'
    
    # Save bvecs into a .bvec file, 3 rows with # directions as columns
    with open(f'{fname}.bvec', 'w') as f:
        for row in s.bvec.T:  # Transpose to match the required format
            f.write(format_str % tuple(row))

# Helper function: get field if exists, return default value otherwise
def try_get_field(s, field, dft_val=None):
    return s.get(field, dft_val)

import numpy as np

def dcm2quat(R):
    """
    Converts a 3x3 rotation matrix R into a quaternion q.

    Parameters:
    R (ndarray): 3x3 rotation matrix.

    Returns:
    q (ndarray): Quaternion [q1, q2, q3, q4].
    proper (int): Sign of the determinant of R.
    """
    proper = np.sign(np.linalg.det(R))  # determinant sign, for checking handedness
    if proper < 0:
        R[:, 2] = -R[:, 2]  # Adjust for left-handed coordinate system

    q = np.sqrt(np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]).dot(np.diag(R)) + 1) / 2
    
    if not np.isreal(q[0]):
        q[0] = 0  # Trace(R) + 1 < 0 case, set to zero

    m, ind = np.max(q), np.argmax(q)

    if ind == 0:
        q[1] = (R[2, 1] - R[1, 2]) / (4 * m)
        q[2] = (R[0, 2] - R[2, 0]) / (4 * m)
        q[3] = (R[1, 0] - R[0, 1]) / (4 * m)
    elif ind == 1:
        q[0] = (R[2, 1] - R[1, 2]) / (4 * m)
        q[2] = (R[0, 1] + R[1, 0]) / (4 * m)
        q[3] = (R[2, 0] + R[0, 2]) / (4 * m)
    elif ind == 2:
        q[0] = (R[0, 2] - R[2, 0]) / (4 * m)
        q[1] = (R[0, 1] + R[1, 0]) / (4 * m)
        q[3] = (R[1, 2] + R[2, 1]) / (4 * m)
    elif ind == 3:
        q[0] = (R[1, 0] - R[0, 1]) / (4 * m)
        q[1] = (R[2, 0] + R[0, 2]) / (4 * m)
        q[2] = (R[1, 2] + R[2, 1]) / (4 * m)

    if q[0] < 0:
        q = -q  # Normalize quaternion as per MRICron standard

    return q, proper

def xform_mat(s, dim):
    """
    Constructs the transformation matrix from the DICOM header information.

    Parameters:
    s (dict): DICOM header with necessary fields.
    dim (tuple): Image dimensions.

    Returns:
    ixyz (ndarray): Orientation information.
    R (ndarray): Transformation matrix from DICOM image space to patient space.
    pixdim (ndarray): Voxel dimensions (PixelSpacing and SliceThickness).
    xyz_unit (int): Unit of measurement (usually 2 for mm).
    """
    # Initialize the rotation matrix from the ImageOrientationPatient field
    R = np.reshape(try_get_field(s, 'ImageOrientationPatient', [1, 0, 0, 0, 1, 0]), (3, 2))
    R = np.column_stack((R, np.cross(R[:, 0], R[:, 1])))  # Right-handed

    foo = np.abs(R)
    ixyz = np.argmax(foo, axis=0) + 1  # Orientation info (permutation of [1, 2, 3])

    if ixyz[1] == ixyz[0]:
        foo[ixyz[1] - 1, 1] = 0
        ixyz[1] = np.argmax(foo[:, 1]) + 1

    if ixyz[2] in ixyz[:2]:
        ixyz[2] = set(range(1, 4)).difference(ixyz[:2]).pop()  # Set difference to find the third index

    if len(dim) < 3 or dim[2] < 2:
        return ixyz, None  # Only single slice, no further processing needed

    iSL = ixyz[2]  # Sagittal, Coronal, or Axial slice
    cosSL = R[iSL - 1, 2]  # Cosine of the slice orientation

    # Extract pixel spacing
    pixdim = np.array(try_get_field(s, 'PixelSpacing', [1, 1]), dtype=float)
    if pixdim.size == 0:
        pixdim = np.array([1, 1], dtype=float)  # Fake pixel size if not available
        xyz_unit = 0
    else:
        xyz_unit = 2  # mm

    # Get slice thickness
    thk = try_get_field(s, 'SpacingBetweenSlices')
    if thk is None:
        thk = try_get_field(s, 'SliceThickness', pixdim[0])
    pixdim = np.append(pixdim, thk)

    # Apply voxel size scaling to the transformation matrix
    R = np.dot(R, np.diag(pixdim))

    # Create transformation matrix (DICOM xform matrix)
    R = np.vstack([R, np.array([0, 0, 0, 1])])
    R[:3, 3] = try_get_field(s, 'ImagePositionPatient', -np.array(dim) / 2)

    return ixyz, R, pixdim, xyz_unit

def csa_header(s, key, dft=None):
    """
    Retrieves a value from the CSA headers in the DICOM dataset.

    Parameters:
    s (dict-like): The DICOM header, typically a pydicom dataset.
    key (str): The field name to search for.
    dft (any, optional): Default value to return if the field is not found.

    Returns:
    val: The value associated with the key in the CSA header, or the default value if not found.
    """
    # Check for the field in CSAImageHeaderInfo
    if hasattr(s, 'CSAImageHeaderInfo') and key in s.CSAImageHeaderInfo:
        return s.CSAImageHeaderInfo[key]

    # Check for the field in CSASeriesHeaderInfo
    if hasattr(s, 'CSASeriesHeaderInfo') and key in s.CSASeriesHeaderInfo:
        return s.CSASeriesHeaderInfo[key]

    # Return the default value if provided, otherwise None
    return dft if dft is not None else None

def asc_header(s, key):
    """
    Extracts a value from the CSA Series Header in a DICOM dataset.
    
    Parameters:
    s (dict-like): DICOM header or dataset (e.g., pydicom Dataset).
    key (str): The key to search for in the CSA header.
    
    Returns:
    val: The value corresponding to the key in the CSA header, or None if not found.
    """
    val = None
    fld = 'CSASeriesHeaderInfo'
    
    # Check if the CSASeriesHeaderInfo field exists
    if not hasattr(s, fld):
        return val

    # Get the appropriate protocol string (MrPhoenixProtocol or MrProtocol)
    if hasattr(s[fld], 'MrPhoenixProtocol'):
        protocol_str = s[fld].MrPhoenixProtocol
    elif hasattr(s[fld], 'MrProtocol'):  # Older DICOM versions
        protocol_str = s[fld].MrProtocol
    else:  # If CSA header decoding fails
        protocol_str = str(s[fld])
        match = re.search(r'ASCCONV BEGIN(.*)ASCCONV END', protocol_str, re.DOTALL)
        if not match:
            return val
        protocol_str = match.group(1)

    # Search for the key, prefixed by a newline character
    key_pattern = f'\n{key}'
    key_idx = protocol_str.find(key_pattern)
    if key_idx == -1:
        return val

    # Move index to the value part after the key
    key_idx += len(key_pattern)
    value_line = protocol_str[key_idx:].split('\n', 1)[0].strip()

    # Extract the value after the '=' sign
    equal_idx = value_line.find('=')
    if equal_idx != -1:
        value_str = value_line[equal_idx + 1:].strip()

        # Determine the type of the value (string, hex, or decimal)
        if value_str.startswith('""'):  # Empty string
            val = value_str[2:-2]
        elif value_str.startswith('"'):  # String parameter
            val = value_str[1:-1]
        elif value_str.startswith('0x'):  # Hexadecimal parameter
            val = int(value_str[2:], 16)
        else:  # Decimal or floating-point parameter
            try:
                val = float(value_str)
            except ValueError:
                val = value_str

    return val

def compress_func(fname):
    """
    Determines the compression function based on the file signature.
    
    Parameters:
    fname (str): The filename to check for compression type.
    
    Returns:
    func (str): 'unzip' for zip files, 'untar' for gz, tgz, tar files, or empty string for others.
    """
    func = ''
    try:
        with open(fname, 'rb') as fid:
            sig = fid.read(2)
        
        if sig == b'PK':  # Zip file signature
            func = 'unzip'
        elif sig == b'\x1f\x8b':  # Gzip signature
            func = 'untar'
    except FileNotFoundError:
        return func
    
    return func

def phase_direction(s):
    """
    Determines the phase encoding direction and phase axis in image reference.
    
    Parameters:
    s (dict): DICOM header or dataset (e.g., pydicom Dataset).
    
    Returns:
    phPos (bool): Phase direction, True for positive, False for negative.
    iPhase (int): Phase axis, 1 for row, 2 for column.
    """
    phPos = None
    iPhase = None
    fld = 'InPlanePhaseEncodingDirection'
    
    if fld in s:
        if s[fld].startswith('COL'):
            iPhase = 2  # Column phase encoding direction
        elif s[fld].startswith('ROW'):
            iPhase = 1  # Row phase encoding direction
        else:
            raise ValueError(f"Unknown {fld} for {s['NiftiName']}: {s[fld]}")

    manufacturer = s['Manufacturer'].upper()
    
    if 'SIEMENS' in manufacturer:
        phPos = csa_header(s, 'PhaseEncodingDirectionPositive')  # Siemens-specific
    elif 'GE' in manufacturer:
        fld = 'ProtocolDataBlock'
        if fld in s and 'VIEWORDER' in s[fld]:
            phPos = s[fld]['VIEWORDER'] == 1  # Bottom-up
    elif 'PHILIPS' in manufacturer:
        if 'ImageOrientationPatient' not in s:
            return phPos, iPhase
        fld = 'MRStackPreparationDirection'
        if 'Stack' in s and fld in s['Stack']['Item_1']:
            R = np.reshape(s['ImageOrientationPatient'], (3, 2))
            ixy = np.argmax(np.abs(R), axis=0)
            d = s['Stack']['Item_1'][fld][0]  # e.g., 'AP'
            
            if iPhase is None:  # No InPlanePhaseEncodingDirection
                iPhase = 'RLAPFH'.find(d) // 2 + 1
                iPhase = (ixy == iPhase).nonzero()[0][0] + 1
            
            if d in 'LPH':
                phPos = False  # Negative phase encoding direction
            elif d in 'RAF':
                phPos = True  # Positive phase encoding direction
            
            if R[ixy[iPhase - 1], iPhase - 1] < 0:
                phPos = not phPos  # Reverse phase direction if necessary
    
    return phPos, iPhase

def gui_callback(h, evt, cmd, fh, hs):
    """
    Handles various GUI actions.
    
    Parameters:
    h: UI component.
    evt: UI event.
    cmd (str): Command to execute.
    fh: Main window handle.
    hs: GUI components (as a dictionary).
    """
    if cmd == 'do_convert':
        src = hs['src'].get()
        dst = hs['dst'].get()
        if not src or not dst:
            print("Source and Result folder must be specified")
            return
        
        rstFmt = (hs['rstFmt'].current() - 1) * 2
        if hs['gzip'].get():
            rstFmt += 1
        if hs['rst3D'].get():
            rstFmt += 4
        
        # Call conversion (dicm2nii equivalent)
        dicm2nii(src, dst, rstFmt)
    
    elif cmd == 'dstDialog':
        folder = filedialog.askdirectory(title="Select a folder for result files")
        if folder:
            hs['dst'].set(folder)
    
    elif cmd == 'srcDir':
        folder = filedialog.askdirectory(title="Select a folder containing convertible files")
        if folder:
            hs['src'].set(folder)
    
    elif cmd == 'srcFile':
        files = filedialog.askopenfilenames(
            title="Select convertible files",
            filetypes=[('Convertible Files', '*.zip *.tgz *.tar *.dcm *.PAR *.HEAD *.fmr *.vmr *.dmr')]
        )
        if files:
            hs['src'].set('; '.join(files))
    
    # Additional callbacks can be added here

def create_gui():
    """
    Creates a basic GUI for the dicm2nii application using tkinter.
    """
    root = Tk()
    root.title("dicm2nii - DICOM to NIfTI Converter")
    root.geometry("420x256")
    
    hs = {}

    # Add source folder or files
    hs['src'] = StringVar()
    hs['dst'] = StringVar()

    src_label = Label(root, text="Source folder/files:")
    src_label.grid(row=0, column=0, padx=8, pady=8, sticky="w")
    
    src_entry = Entry(root, textvariable=hs['src'], width=40)
    src_entry.grid(row=0, column=1, padx=8)
    
    dst_label = Label(root, text="Result folder:")
    dst_label.grid(row=1, column=0, padx=8, pady=8, sticky="w")
    
    dst_entry = Entry(root, textvariable=hs['dst'], width=40)
    dst_entry.grid(row=1, column=1, padx=8)
    
    # Output format and options
    hs['rstFmt'] = ttk.Combobox(root, values=[" .nii", " .hdr/.img"])
    hs['rstFmt'].grid(row=2, column=1, padx=8)
    
    hs['gzip'] = IntVar()
    gzip_check = Checkbutton(root, text="Compress", variable=hs['gzip'])
    gzip_check.grid(row=2, column=2, padx=8)

    hs['rst3D'] = IntVar()
    rst3D_check = Checkbutton(root, text="SPM 3D", variable=hs['rst3D'])
    rst3D_check.grid(row=2, column=3, padx=8)
    
    # Start conversion button
    convert_button = Button(root, text="Start conversion", command=lambda: gui_callback(None, None, 'do_convert', root, hs))
    convert_button.grid(row=3, column=1, padx=8, pady=20)
    
    root.mainloop()

def multiFrameFields(s):
    """
    Process multi-frame DICOM fields in the DICOM header.
    
    Parameters:
    s (dict-like): DICOM header (e.g., pydicom Dataset).
    
    Returns:
    s (dict-like): Updated DICOM header with additional fields.
    """
    pffgs = 'PerFrameFunctionalGroupsSequence'
    if not all(k in s for k in ['SharedFunctionalGroupsSequence', pffgs]):
        return s

    # List of fields to check and assign
    flds = ['EchoTime', 'PixelSpacing', 'SpacingBetweenSlices', 'SliceThickness',
            'RepetitionTime', 'FlipAngle', 'RescaleIntercept', 'RescaleSlope',
            'ImageOrientationPatient', 'ImagePositionPatient',
            'InPlanePhaseEncodingDirection']
    
    for fld in flds:
        if fld not in s:
            a = MF_val(fld, s, 1)
            if a is not None:
                s[fld] = a

    if 'EchoTime' not in s:
        a = MF_val('EffectiveEchoTime', s, 1)
        if a is not None:
            s['EchoTime'] = a
        elif 'EchoTimeDisplay' in s:
            s['EchoTime'] = s['EchoTimeDisplay']

    nFrame = try_get_field(s, 'NumberOfFrames', default=len(s[pffgs]['FrameStart']))

    # Check ImageOrientationPatient consistency for 1st and last frame
    fld = 'ImageOrientationPatient'
    val = MF_val(fld, s, nFrame)
    if val is not None and fld in s and np.any(np.abs(val - s[fld]) > 1e-4):
        return None  # Inconsistent orientation, silently ignore

    # Handle last frame values for specified fields
    flds_last = ['DiffusionDirectionality', 'ImagePositionPatient',
                 'ComplexImageComponent', 'RescaleIntercept', 'RescaleSlope']
    
    for fld in flds_last:
        a = MF_val(fld, s, nFrame)
        if a is not None:
            s['LastFile'][fld] = a

    # Check if 2nd frame matches 1st for ImagePositionPatient
    fld = 'ImagePositionPatient'
    val = MF_val(fld, s, 2)
    if val is not None and fld in s and np.all(np.abs(s[fld] - val) < 1e-4):
        s['Dim3IsVolume'] = True

    # Handle LocationsInAcquisition and slice ordering
    if 'LocationsInAcquisition' not in s:
        s2 = {'ImagePositionPatient': np.full((3, nFrame), np.nan)}
        s2 = dicm_hdr(s, s2, range(1, nFrame + 1))
        iSL = xform_mat(s)
        ipp = s2['ImagePositionPatient'][iSL[2], :]
        err, s['LocationsInAcquisition'], sliceN = check_image_position(ipp)
        if err:
            errorLog(f"{err} for '{s['Filename']}'. Series skipped.")
            return None

    # Handle weird slice ordering (only seen in Philips PAR files)
    nSL = len(s['LocationsInAcquisition'])
    i = MF_val('SliceNumberMR', s, 1)
    if i is not None:
        i2 = MF_val('SliceNumberMR', s, nFrame)
        if i == [1, nSL] or i == [nSL, 1]:
            return s

    return s

def MF_val(fld, s, iFrame):
    """
    Return a value from Shared or PerFrame FunctionalGroupsSequence.

    Parameters:
    fld (str): The field to extract.
    s (dict-like): DICOM header.
    iFrame (int): Frame index.

    Returns:
    val: The extracted value or None if not found.
    """
    sequence_map = {
        'EffectiveEchoTime': 'MREchoSequence',
        'DiffusionDirectionality': 'MRDiffusionSequence',
        'ComplexImageComponent': 'MRImageFrameTypeSequence',
        'RepetitionTime': 'MRTimingAndRelatedParametersSequence',
        'ImagePositionPatient': 'PlanePositionSequence',
        'ImageOrientationPatient': 'PlaneOrientationSequence',
        'PixelSpacing': 'PixelMeasuresSequence',
        'RescaleIntercept': 'PixelValueTransformationSequence',
        'InPlanePhaseEncodingDirection': 'MRFOVGeometrySequence',
        'SliceNumberMR': 'PrivatePerFrameSq'
    }
    
    sq = sequence_map.get(fld, None)
    if sq is None:
        raise ValueError(f"Sequence for {fld} not set.")
    
    pffgs = 'PerFrameFunctionalGroupsSequence'
    try:
        return s['SharedFunctionalGroupsSequence']['Item_1'][sq]['Item_1'][fld]
    except KeyError:
        try:
            return s[pffgs][f'Item_{iFrame}'][sq]['Item_1'][fld]
        except KeyError:
            return None

def split_philips_phase(nii, s):
    """
    Splits Philips NIfTI file into magnitude and phase components.

    Parameters:
    nii (Nifti1Image): NIfTI image.
    s (dict-like): DICOM header.

    Returns:
    nii (Nifti1Image): Updated NIfTI image (magnitude component).
    niiP (Nifti1Image): Phase component (if applicable).
    """
    niiP = None
    if try_get_field(s, 'ComplexImageComponent', '') != 'MIXED' and (
            'VolumeIsPhase' not in s or all(s['VolumeIsPhase']) or not any(s['VolumeIsPhase'])):
        return nii, niiP

    if 'VolumeIsPhase' not in s:
        dim = nii.header['dim'][3:5]
        iFrames = range(1, dim[1] + 1) if try_get_field(s, 'Dim3IsVolume') else range(1, dim[0] * dim[1] + 1)
        flds = ['PerFrameFunctionalGroupsSequence', 'MRImageFrameTypeSequence', 'ComplexImageComponent']

        if dim[1] == 2:
            iFrames = [dim[0] * dim[1]]
            s1 = {flds[0]: s[flds[0]]}
        else:
            dict = dicm_dict(s['Manufacturer'], flds)
            s1 = dicm_hdr(s['Filename'], dict, iFrames)

        s['VolumeIsPhase'] = np.array([s1[flds[0]][f'Item_{i}'][flds[1]]['Item_1'][flds[2]] == 'PHASE'
                                       for i in range(len(iFrames))])

    niiP = nii.copy()
    niiP.dataobj = nii.dataobj[..., s['VolumeIsPhase']]
    niiP.header['dim'][4] = sum(s['VolumeIsPhase'])

    nii.dataobj = nii.dataobj[..., ~s['VolumeIsPhase']]
    nii.header['dim'][4] = sum(~s['VolumeIsPhase'])

    return nii, niiP

def errorLog(errInfo, folder=None):
    """
    Logs error information to a file.

    Parameters:
    errInfo (str): The error message.
    folder (str): The folder to store the error log.
    """
    if not errInfo:
        return

    if folder is None:
        folder = os.getcwd()

    log_file = os.path.join(folder, 'dicm2nii_warningMsg.txt')

    with open(log_file, 'a') as f:
        f.write(f"{errInfo}\n")

def reviseDate(mfile=None):
    """
    Get the last date string in the revision history.

    Parameters:
    mfile (str): The filename of the script (optional).

    Returns:
    dStr (str): The revision date.
    """
    if mfile is None:
        mfile = __file__  # Use the current file by default

    dStr = '160408?'
    try:
        with open(mfile, 'r') as f:
            content = f.read()

        dates = re.findall(r'% (\d{6}) ', content)
        if dates:
            dStr = dates[-1]

    except FileNotFoundError:
        pass

    return dStr

def csa2pos(s, nSL):
    """
    Computes the image orientation and position for Siemens data.
    
    Parameters:
    s (dict-like): DICOM header.
    nSL (int): Number of slices.
    
    Returns:
    s (dict-like): Updated DICOM header.
    """
    ori = ['Sag', 'Cor', 'Tra']
    sNormal = np.zeros(3)
    
    # Extract normal vector for each orientation
    for i in range(3):
        a = asc_header(s, f"sSliceArray.asSlice[0].sNormal.d{ori[i]}")
        if a is not None:
            sNormal[i] = a

    if np.all(sNormal == 0):
        return s  # No useful info, return

    isMos = try_get_field(s, 'isMos', False)
    revNum = asc_header(s, 'sSliceArray.ucImageNumb') is not None
    cosSL, iSL = np.max(np.abs(sNormal)), np.argmax(np.abs(sNormal))

    if isMos and 'CSAImageHeaderInfo' not in s or 'SliceNormalVector' not in s.get('CSAImageHeaderInfo', {}):
        a = sNormal if not revNum else -sNormal
        s.setdefault('CSAImageHeaderInfo', {})['SliceNormalVector'] = a

    pos = np.zeros((3, 2))
    sl = [0, nSL - 1]

    for j in range(2):
        key = f"sSliceArray.asSlice[{sl[j]}].sPosition.d"
        for i in range(3):
            a = asc_header(s, f"{key}{ori[i]}")
            if a is not None:
                pos[i, j] = a

    if 'SpacingBetweenSlices' not in s:
        if np.all(pos[:, 1] == 0):  # Mprage case
            a = asc_header(s, 'sSliceArray.asSlice[0].dThickness') / nSL
            if a is not None:
                s['SpacingBetweenSlices'] = a
        else:
            s['SpacingBetweenSlices'] = abs(pos[iSL, 1] - pos[iSL, 0]) / (nSL - 1) / cosSL

    if 'PixelSpacing' not in s:
        a = asc_header(s, 'sSliceArray.asSlice[0].dReadoutFOV')
        a = a / asc_header(s, 'sKSpace.lBaseResolution')
        interp = asc_header(s, 'sKSpace.uc2DInterpolation')
        if interp:
            a = a / 2
        if a is not None:
            s['PixelSpacing'] = np.array([a, a])

    # Calculate image orientation
    R = np.zeros((3, 3))
    R[:, 2] = sNormal
    if 'ImageOrientationPatient' in s:
        R[:, 0:2] = np.reshape(s['ImageOrientationPatient'], (3, 2))
    else:
        # Calculate R based on the slice index
        if iSL == 2:  # Axial slice
            R[:, 1] = [0, R[2, 2], -R[1, 2]] / np.sqrt(R[1:3, 2] @ R[1:3, 2])
            R[:, 0] = np.cross(R[:, 1], R[:, 2])
        elif iSL == 1:  # Coronal slice
            R[:, 0] = [R[1, 2], -R[0, 2], 0] / np.sqrt(R[0:2, 2] @ R[0:2, 2])
            R[:, 1] = np.cross(R[:, 2], R[:, 0])
        elif iSL == 0:  # Sagittal slice
            R[:, 0] = [-R[1, 2], R[0, 2], 0] / np.sqrt(R[0:2, 2] @ R[0:2, 2])
            R[:, 1] = np.cross(R[:, 0], R[:, 2])

        # Apply in-plane rotation
        rot = asc_header(s, 'sSliceArray.asSlice[0].dInPlaneRot')
        if rot is None:
            rot = 0
        rot = rot - round(rot / np.pi * 2) * np.pi / 2
        ca, sa = np.cos(rot), np.sin(rot)
        R = np.dot(R, np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]]))
        s['ImageOrientationPatient'] = R[:, 0:2].flatten()

    # Handle ImagePositionPatient
    if 'ImagePositionPatient' not in s:
        dim = np.array([s['Columns'], s['Rows']], dtype=float)
        if np.all(pos[:, 1] == 0):
            if not all(f in s for f in ['PixelSpacing', 'SpacingBetweenSlices']):
                return s
            R = R @ np.diag(np.concatenate([s['PixelSpacing'], [s['SpacingBetweenSlices']]]))
            x = np.array([[-dim[0] / 2, -dim[1] / 2], [-(nSL - 1) / 2, (nSL - 1) / 2]])
            pos = R @ x + pos[:, 0].reshape(-1, 1)
        else:
            if 'PixelSpacing' not in s:
                return s
            R = R[:, 0:2] @ np.diag(s['PixelSpacing'])
            pos = pos - R @ (dim / 2).reshape(-1, 1)

        if revNum:
            pos[:, [0, 1]] = pos[:, [1, 0]]
        if isMos:
            pos[:, 1] = pos[:, 0]

        s['ImagePositionPatient'] = pos[:, 0]
        s['LastFile'] = {'ImagePositionPatient': pos[:, 1]}

    return s

def useParTool():
    """
    Check if parallel processing tools are available and initiate if necessary.

    Returns:
    doParal (bool): True if parallel processing is available and initialized.
    """
    doParal = multiprocessing.cpu_count() > 1

    if doParal:
        try:
            pool = multiprocessing.get_context('fork').Pool()
            pool.close()
        except Exception as e:
            print(f"Error initializing parallel pool: {e}")
            doParal = False

    return doParal

def set_nii_ext(s, save_pn):
    """
    Generate NIfTI extension text from DICOM header fields.

    Parameters:
    s (dict-like): DICOM header or dataset.
    save_pn (bool): Whether to save patient name in the extension.

    Returns:
    ext (dict): Dictionary with NIfTI extension data.
    """
    flds = [
        'NiftiCreator', 'SeriesNumber', 'SeriesDescription', 'ImageType', 'Modality',
        'AcquisitionDateTime', 'bval', 'bvec', 'ReadoutSeconds', 'SliceTiming',
        'UnwarpDirection', 'EffectiveEPIEchoSpacing', 'EchoTime', 'deltaTE',
        'PatientName', 'PatientSex', 'PatientAge', 'PatientSize', 'PatientWeight',
        'PatientPosition', 'SliceThickness', 'FlipAngle', 'RBMoCoTrans', 'RBMoCoRot',
        'Manufacturer', 'SoftwareVersion', 'MRAcquisitionType', 'InstitutionName',
        'ScanningSequence', 'SequenceVariant', 'ScanOptions', 'SequenceName'
    ]

    if not save_pn:
        flds.remove('PatientName')

    ext = {'ecode': 6, 'edata': ''}  # NIfTI extension type 6: text

    for fld in flds:
        val = try_get_field(s, fld)
        if val is None:
            continue

        if isinstance(val, str):
            ext_data = f"'{val}'"
        elif np.isscalar(val):
            ext_data = f"{val:.8g}"
        elif isinstance(val, (list, np.ndarray)) and len(val) == 1:
            ext_data = f"{val[0]:.8g}"
        elif isinstance(val, (list, np.ndarray)):
            if val.ndim == 1:  # Vector
                ext_data = f"[{' '.join(f'{x:.8g}' for x in val)}]"
            else:  # Matrix
                ext_data = f"[{' '.join(' '.join(f'{x:.8g}' for x in row) for row in val)}]"
        else:
            continue  # Skip unsupported types

        ext['edata'] += f"{fld} = {ext_data};\n"

    return ext

def fix_multiband_slice_timing(nSL, mb):
    """
    Fix broken multiband slice timing for even nShot cases.

    Parameters:
    nSL (int): Number of slices.
    mb (int): Multiband factor.

    Returns:
    timing (list): Corrected slice timing order.
    """
    nShot = nSL // mb
    inc = 3  # Increment factor

    # Generate corrected slice timing order
    timing = []
    for i in range(nShot):
        timing.extend([i + j * nShot for j in range(mb)])

    return timing

def mb_slicetiming(s, TA):
    """
    Corrects multiband slice timing for Siemens DICOM data.
    
    Parameters:
    s (dict-like): DICOM header.
    TA (float): Acquisition time.
    
    Returns:
    t (np.array): Corrected slice timing.
    """
    dict = dicm_dict(s['Manufacturer'], 'MosaicRefAcqTimes')
    s2 = dicm_hdr(s['LastFile']['Filename'], dict)
    t = s2.get('MosaicRefAcqTimes', None)  # Try last volume first

    if t is None:
        return None

    nSL = int(s['LocationsInAcquisition'])
    mb = int(np.ceil((max(t) - min(t)) / TA))  # Based on incorrect timing pattern
    if mb == 1 or nSL % mb > 0:
        return t  # Not multiband or incorrect guess

    nShot = nSL // mb
    ucMode = asc_header(s, 'sSliceArray.ucMode')  # 1/2/4: Asc/Desc/Inter
    if ucMode is None:
        return t

    t = np.linspace(0, TA, nShot + 1)[:-1]  # Default ascending mode (ucMode == 1)
    t = np.tile(t, mb)

    if ucMode == 2:  # Descending
        t = t[::-1]
    elif ucMode == 4:  # Interleaved
        if nShot % 2:  # Odd number of shots
            inc = 2
        else:
            inc = nShot // 2 - 1
            if inc % 2 == 0:
                inc -= 1
            errorLog(f"{s['NiftiName']}: multiband interleaved order, even number of shots. "
                     "The SliceTiming information may be incorrect.")

        ind = np.mod(np.arange(nShot) * inc, nShot)
        t = np.zeros(nSL)
        for i in range(nShot):
            t[ind[i]::nShot] = (i / nShot) * TA

    if csa_header(s, 'ProtocolSliceNumber') > 0:
        t = t[::-1]  # Reverse number

    return t

def checkImagePosition(ipp):
    """
    Checks ImagePositionPatient for multiple slices/volumes.
    
    Parameters:
    ipp (np.array): Image position array.
    
    Returns:
    err (str): Error message if any.
    nSL (int): Number of slices.
    sliceN (np.array): Slice numbers.
    isTZ (bool): Whether Philips XYTZ mode is detected.
    """
    a = np.diff(np.sort(ipp))
    del_err = np.mean(a) * 0.02  # Allow 2% error

    nSL = np.sum(a > del_err) + 1
    err = ''
    sliceN = []
    isTZ = False
    nVol = len(ipp) / nSL

    if nVol % 1 != 0:
        err = 'Missing file(s) detected'
        return err, nSL, sliceN, isTZ

    if nSL < 2:
        return err, nSL, sliceN, isTZ

    isTZ = nVol > 1 and np.all(np.abs(np.diff(ipp[:nVol])) < del_err)
    if isTZ:
        a = ipp[::nVol]
        b = ipp.reshape(nVol, nSL)
    else:
        a = ipp[:nSL]
        b = ipp.reshape(nSL, nVol).T

    _, sliceN = np.argsort(a), np.argsort(a)
    if np.any(np.abs(np.diff(a, 2)) > del_err):
        err = 'Inconsistent slice spacing'
        return err, nSL, sliceN, isTZ

    if nVol > 1:
        b = np.diff(b, axis=0)
        if np.any(np.abs(b.flatten()) > del_err):
            err = 'Irregular slice order'

    return err, nSL, sliceN, isTZ

def save_json(s, fname):
    """
    Saves DICOM metadata into a JSON file.
    
    Parameters:
    s (dict-like): DICOM metadata.
    fname (str): Filename to save the JSON file.
    """
    flds = [
        'NiftiCreator', 'SeriesNumber', 'SeriesDescription', 'ImageType', 'Modality',
        'AcquisitionDateTime', 'bval', 'bvec', 'ReadoutSeconds', 'SliceTiming', 'RepetitionTime',
        'UnwarpDirection', 'EffectiveEPIEchoSpacing', 'EchoTime', 'SecondEchoTime',
        'PatientName', 'PatientSex', 'PatientAge', 'PatientSize', 'PatientWeight',
        'PatientPosition', 'SliceThickness', 'FlipAngle', 'RBMoCoTrans', 'RBMoCoRot',
        'Manufacturer', 'SoftwareVersion', 'MRAcquisitionType', 'InstitutionName',
        'ScanningSequence', 'SequenceVariant', 'ScanOptions', 'SequenceName'
    ]

    json_dict = {}

    for fld in flds:
        if fld in s:
            val = s[fld]
            if fld == 'RepetitionTime':
                val /= 1000  # Convert to seconds
            elif fld == 'UnwarpDirection':
                fld = 'PhaseEncodingDirection'
                if val.startswith('-'):
                    val = val[1] + '-'
            elif fld == 'EffectiveEPIEchoSpacing':
                fld = 'EffectiveEchoSpacing'
                val /= 1000
            elif fld == 'ReadoutSeconds':
                fld = 'TotalReadoutTime'
            elif fld == 'SliceTiming':
                val = (0.5 - val) * s['RepetitionTime'] / 1000  # Convert to FSL style in seconds
            elif fld == 'SecondEchoTime':
                fld = 'EchoTime2'
                val /= 1000
            elif fld == 'EchoTime':
                if 'SecondEchoTime' in s:
                    fld = 'EchoTime1'
                val /= 1000

            json_dict[fld] = val

    with open(f"{fname}.json", 'w') as f:
        json.dump(json_dict, f, indent=4)
        
def checkUpdate(mfile):
    """
    Checks for a newer version of the package on MathWorks File Exchange.
    
    Parameters:
    mfile (str): The filename to check.
    """
    webUrl = 'http://www.mathworks.com/matlabcentral/fileexchange/42997'
    
    try:
        response = requests.get(webUrl)
        response.raise_for_status()

        # Extract latest version date
        latestStr = re.search(r'class="date">.*<td>(\d{4}\.\d{2}\.\d{2})</td>', response.text).group(1)
        latestNum = datetime.strptime(latestStr, "%Y.%m.%d").date()
    except Exception as e:
        print(f"Error checking update: {e}")
        os.system(f"open {webUrl}")  # Open in default browser
        return

    myFileDate = datetime.strptime(reviseDate(mfile), '%Y.%m.%d').date()

    if myFileDate >= latestNum:
        print(f"{mfile} is up to date.")
        return

    print(f"A newer version ({latestStr}) is available. Your version is {myFileDate.strftime('%Y.%m.%d')}.")
    answer = input("Update to the new version? [Yes/No]: ").strip().lower()

    if answer == 'yes':
        fileUrl = f'{webUrl}?controller=file_infos&download=true'
        zipFileName = os.path.join(os.path.dirname(mfile), 'dicm2nii.zip')
        try:
            response = requests.get(fileUrl)
            with open(zipFileName, 'wb') as f:
                f.write(response.content)
            print("Downloaded update.")
            # Perform unzip and update
        except Exception as e:
            print(f"Error downloading update: {e}")

def check_mosaic(s):
    """
    Checks if the DICOM file is a Siemens mosaic and returns the number of images in the mosaic.
    
    Parameters:
    s (dict-like): DICOM metadata.
    
    Returns:
    nMos (int): Number of images in the mosaic.
    """
    # Try to get the number of images in the mosaic from the CSA header
    nMos = csa_header(s, 'NumberOfImagesInMosaic')
    if nMos and nMos > 1:
        return nMos  # Found and confirmed mosaic

    # Siemens bug workaround: check EchoColumnPosition and slice dimensions
    echo_pos = csa_header(s, 'EchoColumnPosition')
    if echo_pos and echo_pos > (s['Columns'] / 2):
        nMos = asc_header(s, 'sSliceArray.lSize')
        if nMos:
            return nMos

    # Last attempt: check for padded zeros in the image data
    image_data = np.fromfile(s['Filename'], dtype='uint16')
    zero_pad = np.sum(image_data == 0)
    if zero_pad > 0:
        return zero_pad

    return 1  # Default to 1 if no mosaic is detected

def nMosaic(s):
    """
    Determines the number of images in the mosaic from the Siemens DICOM file.
    
    Parameters:
    s (dict-like): DICOM header.

    Returns:
    nMos (int): Number of images in the mosaic.
    """
    # Try to get the number of images in the mosaic from CSA header
    nMos = csa_header(s, 'NumberOfImagesInMosaic')
    if nMos is not None and nMos != 0:  # Handle cases where nMos is seen as 0
        return nMos

    # Handle Siemens non-mosaic or improperly labeled mosaic (older Siemens syngo MR versions)
    res = csa_header(s, 'EchoColumnPosition')
    if res is not None:
        res *= 2
        dim = max(s['Columns'], s['Rows'])
        interp = asc_header(s, 'sKSpace.uc2DInterpolation')

        if interp:
            dim /= 2

        if dim / res >= 2:  # Likely to be a mosaic
            a = asc_header(s, 'sSliceArray.lSize')
            if a is not None:
                nMos = a
            return nMos

    # Check for mosaic labeled in ImageType without CSA header
    if not isType(s, '\\MOSAIC'):
        return None  # Non-Siemens, return here

    nMos = tryGetField(s, 'LocationsInAcquisition')
    if nMos is not None:
        return nMos

    # Peek into image data to figure out nMos
    dim = np.array([s['Columns'], s['Rows']], dtype=int)
    img = dicm_img(s, 0) != 0  # Binary mask of non-zero pixels
    b = img[dim[0] - 2:dim[0], dim[1] - 2:dim[1]]  # Bottom-right 2x2 corner

    if np.all(b == 0):  # Padded slice at bottom-right corner
        t = img[0:2, dim[1] - 2:dim[1]]  # Top-right 2x2 corner
        if np.all(t == 0):  # All right tiles padded
            ln = np.sum(img, axis=0)  # Use all rows to determine
        else:  # Use bottom rows to determine
            ln = np.sum(img[dim[0] - 4:dim[0], :], axis=0)
        z = np.where(ln != 0)[0][-1]
        nMos = dim[1] / (dim[1] - z)
        if nMos.is_integer() and dim[0] % nMos == 0:
            nMos = int(nMos)
            return nMos ** 2

    # If no padded slice or previous method fails
    ln = np.sum(img, axis=1) == 0  # Check zeros along the phase encoding direction
    if np.sum(ln) < 2:  # Phase encoding direction is ROW
        ln = np.sum(img, axis=0) == 0
        i = np.where(ln == 0)[0][-1]  # Last non-zero column
        ln[i + 2:] = 0  # Trim to keep only one padded slice

    nMos = np.sum(ln)
    if nMos > 1 and np.all(dim % nMos == 0) and np.all(np.diff(np.where(ln)[0], n=2) == 0):
        return nMos ** 2

    # Last attempt, check NumberOfPhaseEncodingSteps
    if 'NumberOfPhaseEncodingSteps' in s:
        nMos = min(dim) // s['NumberOfPhaseEncodingSteps']
        if nMos > 1 and nMos.is_integer() and np.all(dim % nMos == 0):
            return nMos ** 2

    # If everything fails
    errorLog(f"{ProtocolName(s)}: NumberOfImagesInMosaic not available.")
    return None

