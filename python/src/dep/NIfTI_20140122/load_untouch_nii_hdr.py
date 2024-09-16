import struct

def load_nii_hdr(fileprefix, machine, filetype):
    if filetype == 2:
        fn = f"{fileprefix}.nii"
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{fileprefix}.nii".')
    else:
        fn = f"{fileprefix}.hdr"
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{fileprefix}.hdr".')

    with open(fn, 'rb') as fid:
        hdr = read_header(fid)

    return hdr

def read_header(fid):
    dsr = {
        'hk': header_key(fid),
        'dime': image_dimension(fid),
        'hist': data_history(fid)
    }

    # For Analyze data format
    if dsr['hist']['magic'] not in ['n+1', 'ni1']:
        dsr['hist']['qform_code'] = 0
        dsr['hist']['sform_code'] = 0

    return dsr

def header_key(fid):
    fid.seek(0)
    hk = {}
    hk['sizeof_hdr'] = struct.unpack('i', fid.read(4))[0]
    hk['data_type'] = struct.unpack('10s', fid.read(10))[0].strip(b'\x00').decode()
    hk['db_name'] = struct.unpack('18s', fid.read(18))[0].strip(b'\x00').decode()
    hk['extents'] = struct.unpack('i', fid.read(4))[0]
    hk['session_error'] = struct.unpack('h', fid.read(2))[0]
    hk['regular'] = struct.unpack('c', fid.read(1))[0].decode()
    hk['dim_info'] = struct.unpack('B', fid.read(1))[0]
    return hk

def image_dimension(fid):
    dime = {}
    dime['dim'] = struct.unpack('8h', fid.read(16))
    dime['intent_p1'] = struct.unpack('f', fid.read(4))[0]
    dime['intent_p2'] = struct.unpack('f', fid.read(4))[0]
    dime['intent_p3'] = struct.unpack('f', fid.read(4))[0]
    dime['intent_code'] = struct.unpack('h', fid.read(2))[0]
    dime['datatype'] = struct.unpack('h', fid.read(2))[0]
    dime['bitpix'] = struct.unpack('h', fid.read(2))[0]
    dime['slice_start'] = struct.unpack('h', fid.read(2))[0]
    dime['pixdim'] = struct.unpack('8f', fid.read(32))
    dime['vox_offset'] = struct.unpack('f', fid.read(4))[0]
    dime['scl_slope'] = struct.unpack('f', fid.read(4))[0]
    dime['scl_inter'] = struct.unpack('f', fid.read(4))[0]
    dime['slice_end'] = struct.unpack('h', fid.read(2))[0]
    dime['slice_code'] = struct.unpack('B', fid.read(1))[0]
    dime['xyzt_units'] = struct.unpack('B', fid.read(1))[0]
    dime['cal_max'] = struct.unpack('f', fid.read(4))[0]
    dime['cal_min'] = struct.unpack('f', fid.read(4))[0]
    dime['slice_duration'] = struct.unpack('f', fid.read(4))[0]
    dime['toffset'] = struct.unpack('f', fid.read(4))[0]
    dime['glmax'] = struct.unpack('i', fid.read(4))[0]
    dime['glmin'] = struct.unpack('i', fid.read(4))[0]
    return dime

def data_history(fid):
    hist = {}
    hist['descrip'] = struct.unpack('80s', fid.read(80))[0].strip(b'\x00').decode()
    hist['aux_file'] = struct.unpack('24s', fid.read(24))[0].strip(b'\x00').decode()
    hist['qform_code'] = struct.unpack('h', fid.read(2))[0]
    hist['sform_code'] = struct.unpack('h', fid.read(2))[0]
    hist['quatern_b'] = struct.unpack('f', fid.read(4))[0]
    hist['quatern_c'] = struct.unpack('f', fid.read(4))[0]
    hist['quatern_d'] = struct.unpack('f', fid.read(4))[0]
    hist['qoffset_x'] = struct.unpack('f', fid.read(4))[0]
    hist['qoffset_y'] = struct.unpack('f', fid.read(4))[0]
    hist['qoffset_z'] = struct.unpack('f', fid.read(4))[0]
    hist['srow_x'] = struct.unpack('4f', fid.read(16))
    hist['srow_y'] = struct.unpack('4f', fid.read(16))
    hist['srow_z'] = struct.unpack('4f', fid.read(16))
    hist['intent_name'] = struct.unpack('16s', fid.read(16))[0].strip(b'\x00').decode()
    hist['magic'] = struct.unpack('4s', fid.read(4))[0].strip(b'\x00').decode()
    return hist

# Example usage:
# hdr = load_nii_hdr('fileprefix', 'little', 2)
