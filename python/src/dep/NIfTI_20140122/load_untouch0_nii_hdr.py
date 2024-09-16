import os
import struct
import nibabel as nib

def load_nii_hdr(fileprefix, machine='native'):
    fn = f"{fileprefix}.hdr"
    try:
        with open(fn, 'rb') as fid:
            hdr = read_header(fid)
    except IOError:
        raise ValueError(f"Cannot open file {fn}.")
    return hdr

def read_header(fid):
    dsr = {
        'hk': header_key(fid),
        'dime': image_dimension(fid),
        'hist': data_history(fid)
    }
    return dsr

def header_key(fid):
    # Read header key
    hk = {}
    hk['sizeof_hdr'] = struct.unpack('i', fid.read(4))[0]
    hk['data_type'] = fid.read(10).decode('utf-8').strip()
    hk['db_name'] = fid.read(18).decode('utf-8').strip()
    hk['extents'] = struct.unpack('i', fid.read(4))[0]
    hk['session_error'] = struct.unpack('h', fid.read(2))[0]
    hk['regular'] = fid.read(1).decode('utf-8').strip()
    hk['hkey_un0'] = fid.read(1).decode('utf-8').strip()
    return hk

def image_dimension(fid):
    # Read image dimension
    dime = {}
    dime['dim'] = struct.unpack('8h', fid.read(16))
    dime['vox_units'] = fid.read(4).decode('utf-8').strip()
    dime['cal_units'] = fid.read(8).decode('utf-8').strip()
    dime['unused1'] = struct.unpack('h', fid.read(2))[0]
    dime['datatype'] = struct.unpack('h', fid.read(2))[0]
    dime['bitpix'] = struct.unpack('h', fid.read(2))[0]
    dime['dim_un0'] = struct.unpack('h', fid.read(2))[0]
    dime['pixdim'] = struct.unpack('8f', fid.read(32))
    dime['vox_offset'] = struct.unpack('f', fid.read(4))[0]
    dime['roi_scale'] = struct.unpack('f', fid.read(4))[0]
    dime['funused1'] = struct.unpack('f', fid.read(4))[0]
    dime['funused2'] = struct.unpack('f', fid.read(4))[0]
    dime['cal_max'] = struct.unpack('f', fid.read(4))[0]
    dime['cal_min'] = struct.unpack('f', fid.read(4))[0]
    dime['compressed'] = struct.unpack('i', fid.read(4))[0]
    dime['verified'] = struct.unpack('i', fid.read(4))[0]
    dime['glmax'] = struct.unpack('i', fid.read(4))[0]
    dime['glmin'] = struct.unpack('i', fid.read(4))[0]
    return dime

def data_history(fid):
    # Read data history
    hist = {}
    hist['descrip'] = fid.read(80).decode('utf-8').strip()
    hist['aux_file'] = fid.read(24).decode('utf-8').strip()
    hist['orient'] = fid.read(1).decode('utf-8')
    hist['originator'] = struct.unpack('5h', fid.read(10))
    hist['generated'] = fid.read(10).decode('utf-8').strip()
    hist['scannum'] = fid.read(10).decode('utf-8').strip()
    hist['patient_id'] = fid.read(10).decode('utf-8').strip()
    hist['exp_date'] = fid.read(10).decode('utf-8').strip()
    hist['exp_time'] = fid.read(10).decode('utf-8').strip()
    hist['hist_un0'] = fid.read(3).decode('utf-8').strip()
    hist['views'] = struct.unpack('i', fid.read(4))[0]
    hist['vols_added'] = struct.unpack('i', fid.read(4))[0]
    hist['start_field'] = struct.unpack('i', fid.read(4))[0]
    hist['field_skip'] = struct.unpack('i', fid.read(4))[0]
    hist['omax'] = struct.unpack('i', fid.read(4))[0]
    hist['omin'] = struct.unpack('i', fid.read(4))[0]
    hist['smax'] = struct.unpack('i', fid.read(4))[0]
    hist['smin'] = struct.unpack('i', fid.read(4))[0]
    return hist

# Example usage
# hdr = load_nii_hdr('example_fileprefix')
