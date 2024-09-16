import struct
import os

def load_nii_hdr(fileprefix):
    if not fileprefix:
        raise ValueError('Usage: [hdr, filetype, fileprefix, machine] = load_nii_hdr(filename)')

    machine = 'ieee-le'
    new_ext = False

    if fileprefix.endswith('.nii'):
        new_ext = True
        fileprefix = fileprefix[:-4]

    if fileprefix.endswith('.hdr'):
        fileprefix = fileprefix[:-4]

    if fileprefix.endswith('.img'):
        fileprefix = fileprefix[:-4]

    if new_ext:
        fn = f'{fileprefix}.nii'
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{fileprefix}.nii".')
    else:
        fn = f'{fileprefix}.hdr'
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{fileprefix}.hdr".')

    with open(fn, 'rb') as fid:
        hdr = read_header(fid, machine)

    if hdr['hk']['sizeof_hdr'] != 348:
        machine = 'ieee-be' if machine == 'ieee-le' else 'ieee-le'
        with open(fn, 'rb') as fid:
            hdr = read_header(fid, machine)
            if hdr['hk']['sizeof_hdr'] != 348:
                raise ValueError(f'File "{fn}" is corrupted.')

    filetype = 2 if hdr['hist']['magic'] == 'n+1' else (1 if hdr['hist']['magic'] == 'ni1' else 0)
    return hdr, filetype, fileprefix, machine

def read_header(fid, machine):
    dsr = {
        'hk': header_key(fid),
        'dime': image_dimension(fid),
        'hist': data_history(fid)
    }

    if dsr['hist']['magic'] not in ['n+1', 'ni1']:
        dsr['hist']['qform_code'] = 0
        dsr['hist']['sform_code'] = 0

    return dsr

def header_key(fid):
    fid.seek(0)
    hk = {
        'sizeof_hdr': struct.unpack('i', fid.read(4))[0],
        'data_type': fid.read(10).decode('utf-8').strip(),
        'db_name': fid.read(18).decode('utf-8').strip(),
        'extents': struct.unpack('i', fid.read(4))[0],
        'session_error': struct.unpack('h', fid.read(2))[0],
        'regular': fid.read(1).decode('utf-8'),
        'dim_info': struct.unpack('B', fid.read(1))[0]
    }
    return hk

def image_dimension(fid):
    dim = struct.unpack('8h', fid.read(16))
    intent_p1, intent_p2, intent_p3 = struct.unpack('3f', fid.read(12))
    intent_code, datatype, bitpix, slice_start = struct.unpack('4h', fid.read(8))
    pixdim = struct.unpack('8f', fid.read(32))
    vox_offset, scl_slope, scl_inter = struct.unpack('3f', fid.read(12))
    slice_end = struct.unpack('h', fid.read(2))[0]
    slice_code = struct.unpack('B', fid.read(1))[0]
    xyzt_units = struct.unpack('B', fid.read(1))[0]
    cal_max, cal_min, slice_duration, toffset = struct.unpack('4f', fid.read(16))
    glmax, glmin = struct.unpack('2i', fid.read(8))
    
    dime = {
        'dim': dim,
        'intent_p1': intent_p1,
        'intent_p2': intent_p2,
        'intent_p3': intent_p3,
        'intent_code': intent_code,
        'datatype': datatype,
        'bitpix': bitpix,
        'slice_start': slice_start,
        'pixdim': pixdim,
        'vox_offset': vox_offset,
        'scl_slope': scl_slope,
        'scl_inter': scl_inter,
        'slice_end': slice_end,
        'slice_code': slice_code,
        'xyzt_units': xyzt_units,
        'cal_max': cal_max,
        'cal_min': cal_min,
        'slice_duration': slice_duration,
        'toffset': toffset,
        'glmax': glmax,
        'glmin': glmin
    }
    return dime

def data_history(fid):
    hist = {
        'descrip': fid.read(80).decode('utf-8').strip(),
        'aux_file': fid.read(24).decode('utf-8').strip(),
        'qform_code': struct.unpack('h', fid.read(2))[0],
        'sform_code': struct.unpack('h', fid.read(2))[0],
        'quatern_b': struct.unpack('f', fid.read(4))[0],
        'quatern_c': struct.unpack('f', fid.read(4))[0],
        'quatern_d': struct.unpack('f', fid.read(4))[0],
        'qoffset_x': struct.unpack('f', fid.read(4))[0],
        'qoffset_y': struct.unpack('f', fid.read(4))[0],
        'qoffset_z': struct.unpack('f', fid.read(4))[0],
        'srow_x': struct.unpack('4f', fid.read(16)),
        'srow_y': struct.unpack('4f', fid.read(16)),
        'srow_z': struct.unpack('4f', fid.read(16)),
        'intent_name': fid.read(16).decode('utf-8').strip(),
        'magic': fid.read(4).decode('utf-8').strip()
    }
    return hist
