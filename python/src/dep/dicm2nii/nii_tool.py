import nibabel as nib
import numpy as np
import gzip
import os

# Global variables to mimic MATLAB persistent variables
C = None
para = None

def nii_tool(cmd, *args):
    global C, para
    if C is None or para is None:
        C, para = niiHeader()
    
    if cmd == 'init':
        return init_nii(args[0], args[1] if len(args) > 1 else None)
    elif cmd == 'save':
        return save_nii(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
    elif cmd == 'hdr':
        return hdr_nii(args[0])
    elif cmd == 'img':
        return img_nii(args[0])
    elif cmd == 'ext':
        return ext_nii(args[0])
    elif cmd == 'load':
        return load_nii(args[0])
    elif cmd == 'cat3D':
        return cat3D_nii(args[0])
    elif cmd == 'RGBStyle':
        return RGBStyle(args[0] if len(args) > 0 else None)
    elif cmd == 'default':
        return default_nii(args)
    elif cmd == 'update':
        return update_nii(args[0])
    else:
        raise ValueError(f"Invalid command for nii_tool: {cmd}")

def niiHeader():
    # Define NIfTI header based on version
    pf = {'version': 1, 'rgb_dim': 1}
    niiVer = pf['version']
    
    if niiVer == 1:
        C = [
            ('sizeof_hdr', 1, 'int32', 348),
            # Add other fields based on the given MATLAB struct
        ]
    elif niiVer == 2:
        C = [
            ('sizeof_hdr', 1, 'int32', 540),
            # Add other fields based on the given MATLAB struct
        ]
    else:
        raise ValueError(f"Nifti version {niiVer} is not supported")
    
    # Data type mappings
    D = [
        ('ubit1', 1, 1, 1),
        ('uint8', 2, 8, 1),
        # Add other types based on the given MATLAB struct
    ]
    
    para = {
        'format': [d[0] for d in D],
        'datatype': [d[1] for d in D],
        'bitpix': [d[2] for d in D],
        'valpix': [d[3] for d in D],
        'rgb_dim': pf['rgb_dim'],
        'version': niiVer
    }
    
    return C, para

def init_nii(img, RGB_dim=None):
    global C
    nii = {'hdr': {field[0]: field[3] for field in C}, 'img': img}
    if len(img.shape) > 8:
        raise ValueError('NIfTI img can have up to 7 dimensions')
    if RGB_dim is not None:
        if RGB_dim < 0 or RGB_dim > 8 or RGB_dim % 1 > 0:
            raise ValueError('Invalid RGB_dim number')
        img = np.transpose(img, [i for i in range(RGB_dim-1)] + [i+1 for i in range(RGB_dim, 8)] + [RGB_dim])
    return update_nii(nii)

def save_nii(nii, filename=None, force_3D=False):
    if not isinstance(nii, dict) or 'hdr' not in nii or 'img' not in nii:
        raise ValueError('nii_tool("save") needs a struct from nii_tool("init") or nii_tool("load") as the second input')

    if filename is None:
        if 'file_name' in nii['hdr']:
            filename = nii['hdr']['file_name']
        else:
            raise ValueError('Provide a valid file name as the third input')
    
    niiVer = para['version']
    if 'version' in nii['hdr']:
        niiVer = nii['hdr']['version']
    if niiVer == 1:
        nii['hdr']['sizeof_hdr'] = 348
    elif niiVer == 2:
        nii['hdr']['sizeof_hdr'] = 540
    else:
        raise ValueError(f"Unsupported NIfTI version: {niiVer}")
    
    nii, fmt = update_nii(nii)
    
    # More code to handle saving nii struct to file using nibabel or other methods
    
    pass

def hdr_nii(filename):
    if not isinstance(filename, str):
        raise ValueError('nii_tool("hdr") needs nii file name as second input')
    
    img = nib.load(filename)
    return img.header

def img_nii(filename_or_hdr):
    if isinstance(filename_or_hdr, str):
        img = nib.load(filename_or_hdr)
    elif isinstance(filename_or_hdr, nib.nifti1.Nifti1Header):
        img = filename_or_hdr
    else:
        raise ValueError('nii_tool("img") needs a file name or hdr struct from nii_tool("hdr") as second input')
    
    return img.get_fdata()

def ext_nii(filename_or_hdr):
    # Extensions handling with nibabel or similar libraries
    pass

def load_nii(filename_or_hdr):
    if isinstance(filename_or_hdr, str):
        img = nib.load(filename_or_hdr)
    elif isinstance(filename_or_hdr, nib.nifti1.Nifti1Header):
        img = filename_or_hdr
    else:
        raise ValueError('nii_tool("load") needs a file name or hdr struct from nii_tool("hdr") as second input')
    
    nii = {
        'hdr': img.header,
        'img': img.get_fdata()
    }
    return nii

def cat3D_nii(filenames):
    # Concatenate 3D files into a 4D dataset using nibabel
    pass

def RGBStyle(style=None):
    styles = {'afni': 1, 'mricron': 3, 'fsl': 4}
    curStyle = styles[para['rgb_dim']]
    if style is None:
        return curStyle
    if isinstance(style, str):
        style = style.lower()
        if style not in styles:
            raise ValueError('Invalid style for RGBStyle')
        style = styles[style]
    if style not in styles.values():
        raise ValueError('nii_tool("RGBStyle") can have 1, 3, or 4 as second input')
    if 'rgb_dim' in para:
        para['rgb_dim'] = style
    return curStyle

def default_nii(args):
    flds = ['version', 'rgb_dim']
    val = {fld: para[fld] for fld in flds}
    if not args:
        return val
    if isinstance(args[0], dict):
        in2 = args[0]
    else:
        in2 = dict(zip(args[0][0::2], args[0][1::2]))
    for fld in in2:
        if fld in flds:
            para[fld] = in2[fld]
    if 'version' in in2 and val['version'] != para['version']:
        C = niiHeader(para['version'])
    return val

def update_nii(nii):
    dim = nii['img'].shape
    ndim = len(dim)
    dim += (1,) * (7 - ndim)
    
    if ndim == 8:
        valpix = dim[7]
        if valpix == 4:
            typ = 'RGBA'
            nii['img'] = nii['img'].astype(np.uint8)
        elif valpix == 3:
            typ = 'RGB'
            if np.max(nii['img']) > 1:
                nii['img'] = nii['img'].astype(np.uint8)
            else:
                nii['img'] = nii['img'].astype(np.float32)
        else:
            raise ValueError('Color dimension must have length of 3 for RGB and 4 for RGBA')
        
        dim = dim[:7]
        ndim = np.max(np.nonzero(dim)[0]) + 1
    elif np.isreal(nii['img']):
        typ = 'real'
        valpix = 1
    else:
        typ = 'complex'
        valpix = 2
    
    imgFmt = nii['img'].dtype.name
    ind = [i for i, fmt in enumerate(para['format']) if fmt == imgFmt and para['valpix'][i] == valpix]
    
    if not ind:
        raise ValueError(f"nii_tool does not support {typ} image of '{imgFmt}' type")
    elif len(ind) > 1:
        raise ValueError(f"Non-unique datatype found for {typ} image of '{imgFmt}' type")
    
    fmt = para['format'][ind[0]]
    nii['hdr']['datatype'] = para['datatype'][ind[0]]
    nii['hdr']['bitpix'] = para['bitpix'][ind[0]]
    nii['hdr']['dim'] = [ndim] + list(dim)
    
    if nii['hdr']['sizeof_hdr'] == 348:
        nii['hdr']['glmax'] = int(np.max(nii['img']))
        nii['hdr']['glmin'] = int(np.min(nii['img']))
    
    return nii, fmt
