import os
import numpy as np
import struct
import zlib
import re
import io
import tempfile
import gzip
import random
from pydicom.dataset import Dataset
from datetime import datetime
from collections import defaultdict


def ch2int32(u8, swap):
    if swap:
        u8 = u8[::-1]
    d = int.from_bytes(u8, byteorder='little' if not swap else 'big', signed=False)
    return d

def ch2int16(u8, swap):
    if swap:
        u8 = u8[::-1]
    d = int.from_bytes(u8, byteorder='little' if not swap else 'big', signed=False)
    return d

def dcm_str(b):
    b = b.rstrip(b'\x00')  # Remove trailing null characters
    ch = b.decode().strip()  # Convert to string and remove trailing spaces
    return ch

def val_len(vr, b, expl, swap):
    len16 = ['AE', 'AS', 'AT', 'CS', 'DA', 'DS', 'DT', 'FD', 'FL', 'IS', 'LO', 'LT', 'PN', 'SH', 'SL', 'SS', 'ST', 'TM', 'UI', 'UL', 'US']
    
    if not expl:  # implicit, length irrelevant to vr (faked as CS)
        n = ch2int32(b[0:4], swap)
        nvr = 4  # bytes of VR
    elif vr in len16:  # length in uint16
        n = ch2int16(b[0:2], swap)
        nvr = 2
    else:  # length in uint32 and skip 2 bytes
        n = ch2int32(b[2:6], swap)
        nvr = 6
    
    if n == 13:
        n = 10  # ugly bug fix for some old dicom file
    
    return n, nvr

def vr2fmt(vr):
    vr_dict = {
        'US': 'uint16',
        'OB': 'uint8',
        'FD': 'double',
        'SS': 'int16',
        'UL': 'uint32',
        'SL': 'int32',
        'FL': 'single',
        'AT': 'uint16',
        'OW': 'uint16',
        'UN': 'uint8'
    }
    return vr_dict.get(vr, '')

def read_val(b, vr, swap):
    info = ''
    if vr in ['DS', 'IS']:
        dat = [float(val) for val in b.decode().split('\\')]
    elif vr in ['AE', 'AS', 'CS', 'DA', 'DT', 'LO', 'LT', 'PN', 'SH', 'ST', 'TM', 'UI', 'UT']:
        dat = dcm_str(b)
    else:
        fmt = vr2fmt(vr)
        if not fmt:
            dat = []
            info = f'Given up: Invalid VR ({vr})'
            return dat, info
        dat = struct.unpack(fmt, b)
        if swap:
            dat = swap_bytes(dat)
    return dat, info

def read_item(b8, i, p):
    dat = []
    name = np.nan
    info = ''
    vr = 'CS'  # vr may be used if implicit

    group = b8[i:i+2]
    i += 2
    swap = p['be'] and group != bytes([2, 0])  # good news: no 0x0200 group
    group = ch2int16(group, swap)
    elmnt = ch2int16(b8[i:i+2], swap)
    i += 2
    tag = group * 65536 + elmnt
    if tag == 4294893581:  # || tag == 4294893789 % FFFEE00D or FFFEE0DD
        i += 4  # skip length
        return dat, name, info, i, tag  # rerurn in case n is not 0

    swap = p['be'] and group != 2
    has_vr = p['expl'] or group == 2
    if has_vr:
        vr = b8[i:i+2].decode()
        i += 2  # 2-byte VR

    n, nvr = val_len(vr, b8[i:i+6], has_vr, swap)
    i += nvr
    if n == 0:
        return dat, name, info, i, tag  # empty val

    # Look up item name in dictionary
    ind = np.where(p['dict']['tag'] == tag)[0]
    if ind.size > 0:
        ind = ind[0]
        name = p['dict']['name'][ind]
        if vr in ['UN', 'OB'] or not has_vr:
            vr = p['dict']['vr'][ind]
    elif tag == 524400:  # in case not in dict
        name = 'Manufacturer'
    elif tag == 131088:  # need TransferSyntaxUID even if not in dict
        name = 'TransferSyntaxUID'
    elif tag == 593936:  # 0x0009 0010 GEIIS not dicom compliant
        i += n
        return dat, name, info, i, tag  # seems n is not 0xffffffff
    elif p['fullHdr']:
        if elmnt == 0:
            i += n
            return dat, name, info, i, tag  # skip GroupLength
        if group % 2:
            name = f'Private_{group:04x}_{elmnt:04x}'
        else:
            name = f'Unknown_{group:04x}_{elmnt:04x}'
        if not has_vr:
            vr = 'UN'  # not in dict, will leave as uint8
    elif n < 4294967295:  # no skip for SQ with length 0xffffffff
        i += n
        return dat, name, info, i, tag

    # compressed PixelData, n can be 0xffffffff
    if not has_vr and n == 4294967295:
        vr = 'SQ'  # best guess
    if n + i > p['iPixelData'] and vr != 'SQ':
        i += n
        return dat, name, info, i, tag  # PixelData or err

    if vr == 'SQ':
        n_end = min(i + n, p['iPixelData'])  # n is likely 0xffff ffff
        dat, info, i = read_sq(b8, i, n_end, p, tag == 1375769136)  # isPerFrameSQ?
    else:
        dat, info = read_val(b8[i:i+n], vr, swap)
        i += n

    return dat, name, info, i, tag

def read_sq(b8, i, nEnd, p, isPerFrameSQ):
    rst = {}
    info = ''
    tag1 = None
    j = 0  # j is SQ Item index

    while i < nEnd:  # loop through multi Item under the SQ
        tag = b8[i+2:i+4] + b8[i:i+2]
        i += 4
        if p['be']:
            tag = tag[2:] + tag[:2]
        tag = ch2int32(tag, False)
        if tag != 4294893568:
            i += 4
            return rst, info, i  # only do FFFE E000, Item
        n = ch2int32(b8[i:i+4], p['be'])
        i += 4  # n may be 0xffff ffff
        n = min(i + n, nEnd)
        j += 1

        if isPerFrameSQ:
            if isinstance(p['iFrames'], str):  # 'all' frames
                if j == 1 and not np.isnan(p['nFrames']):
                    rst['FrameStart'] = [np.nan] * p['nFrames']
                rst['FrameStart'][j - 1] = i - 9
            elif j == 1:
                i0 = i - 8  # always read 1st frame, save i0 in case of re-do
            elif j == 2:
                if np.isnan(p['nFrames']) or tag1 is None:
                    p['iFrames'] = 'all'
                    rst = {}
                    j = 0
                    i = i0
                    continue  # re-do the SQ
                tag1_bytes = struct.pack('>I', tag1)
                tag1_bytes = tag1_bytes[2:] + tag1_bytes[:2]
                if p['be'] and tag1_bytes[:2] != b'\x02\x00':
                    tag1_bytes = tag1_bytes[::-1]
                tag1_str = tag1_bytes.decode()
                ind = b8[i0:p['iPixelData']].find(tag1_str.encode()) + i0 - 1
                ind = [x for x in ind if x % 2 == 1]
                nInd = len(ind)
                if nInd != p['nFrames']:
                    tag1PerF = nInd / p['nFrames']
                    if tag1PerF % 1 > 0:
                        p['iFrames'] = 'all'
                        rst = {}
                        j = 0
                        i = i0
                        print('Failed to determine indices for frames. Reading all frames. Maybe slow ...')
                        continue
                    elif tag1PerF > 1:
                        ind = ind[::int(tag1PerF)]
                        nInd = p['nFrames']
                rst['FrameStart'] = [x - 9 for x in ind]
                iItem = 2
                iFrame = list(set([1, 2] + [round(x) for x in p['iFrames']] + [nInd]))
            else:
                iItem += 1
                j = iFrame[iItem]
                i = ind[j]
                n = nEnd

        item_n = f'Item_{j}'
        while i < n:  # loop through multi tags under one Item
            dat, name, info, i, tag = read_item(b8, i, p)
            if tag == 4294893581:
                break  # FFFE E00D ItemDelimitationItem
            if tag1 is None:
                tag1 = tag  # first detected tag for PerFrameSQ
            if dat is None or isinstance(name, float):
                continue  # 0-length or skipped
            if item_n not in rst:
                rst[item_n] = {}
            rst[item_n][name] = dat

    return rst, info, i

def read_csa(csa):
    b = csa
    if len(b) < 4 or b[:4] != b'SV10':
        return csa  # no op if not SV10
    
    chDat = ['AE', 'AS', 'CS', 'DA', 'DT', 'LO', 'LT', 'PN', 'SH', 'ST', 'TM', 'UI', 'UN', 'UT']
    i = 8  # 'SV10' 4 3 2 1
    try:
        nField = ch2int32(b, i)
        i += 8
        rst = {}
        for j in range(nField):
            i += 68  # name(64) and vm(4)
            vr = b[i:i+2].decode('ascii')
            i += 8  # vr(4), syngodt(4)
            n = ch2int32(b, i)
            i += 8
            if n < 1:
                continue  # skip name decoding, faster
            
            nam = b[i-84:i-20].decode('ascii').split('\x00')[0]
            dat = []
            for k in range(n):  # n is often 6, but often only the first contains value
                len = ch2int32(b, i)
                i += 16
                if len < 1:
                    i += (n - k) * 16
                    break  # rest are empty too
                
                foo = b[i:i+len-1].decode('ascii')
                i += (len + 3) // 4 * 4  # multiple 4-byte
                
                if vr not in chDat:
                    tmp = float(foo) if foo.strip() else None
                    if tmp is not None:
                        dat.append(tmp)
                else:
                    dat.append(foo.strip())
            
            if isinstance(dat[0], str):
                dat = [d for d in dat if d]
                if not dat:
                    continue
                if len(dat) == 1:
                    dat = dat[0]
            else:
                dat = dat  # Already a list of floats
            
            rst[nam] = dat
        csa = rst
    except:
        pass  # in case of error, return the original csa
    
    return csa

def read_ProtocolDataBlock(ch):
    n = struct.unpack('i', ch[:4])[0] + 4  # nBytes, zeros may be padded to make 4x
    if ch[4:6] != b'\x1f\x8b' or n > len(ch):  # gz signature
        return ch

    b = zlib.decompress(ch[4:n])
    b = re.findall(r'(\w*)\s+"(.*)"', b.decode('utf-8'), re.DOTALL)
    if not b:
        return ch  # gunzip failed or wrong format

    rst = {}
    for item in b:
        key, value = item
        try:
            a = float(value)
            rst[key] = a
        except ValueError:
            rst[key] = value.strip()

    return rst

def gunzip_mem(gz_bytes):
    bytes_out = b''
    try:
        # Attempt to decompress using gzip and io libraries
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz:
            bytes_out = gz.read()
    except Exception as e:
        try:
            # Create a temporary file to write the gzipped bytes
            tmp_fd, tmp_gz_path = tempfile.mkstemp(suffix='.gz')
            with os.fdopen(tmp_fd, 'wb') as tmp_file:
                tmp_file.write(gz_bytes)
            
            # Decompress the file
            tmp_unzipped_path = tmp_gz_path[:-3]  # remove .gz extension
            with gzip.open(tmp_gz_path, 'rb') as f_in:
                with open(tmp_unzipped_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Read the decompressed bytes
            with open(tmp_unzipped_path, 'rb') as f:
                bytes_out = f.read()
            
            # Clean up the temporary files
            os.remove(tmp_gz_path)
            os.remove(tmp_unzipped_path)
        except Exception as inner_e:
            print(f"Failed to decompress using fallback method: {inner_e}")
    
    return bytes_out

def get_nFrames(s, p):
    if 'NumberOfFrames' in s:
        p['nFrames'] = s['NumberOfFrames']  # useful for PerFrameSQ
    elif all(k in s for k in ('Columns', 'Rows', 'BitsAllocated')) and p['bytes'] < 4294967295:
        spp = float(s.get('SamplesPerPixel', 1))
        n = p['bytes'] * 8 / float(s['BitsAllocated'])
        p['nFrames'] = n / (spp * float(s['Columns']) * float(s['Rows']))
    else:
        p['nFrames'] = float('nan')
    return p

def search_MF_val(s, s1, iFrame):
    """
    Args:
        s: pydicom dataset for a multiframe DICOM
        s1: a dictionary with fields to search and initial values like zeros or NaNs.
            The number of rows indicate the number of values for the tag, and columns for frames indicated by iFrame.
        iFrame: list of frame indices, length consistent with columns of s1 field values.
    """
    if not hasattr(s, 'PerFrameFunctionalGroupsSequence'):
        return s1

    expl = False
    be = False
    if hasattr(s, 'TransferSyntaxUID'):
        expl = s.TransferSyntaxUID != '1.2.840.10008.1.2'
        be = s.TransferSyntaxUID == '1.2.840.10008.1.2.2'

    fStart = [fg.FrameStart for fg in s.PerFrameFunctionalGroupsSequence]
    with open(s.filename, 'rb') as f:
        b0 = f.read(fStart[0])
        f.seek(fStart[0])
        b = f.read(s.PixelData.Start - fStart[0])

    fStart.append(s.PixelData.Start)
    fStart = [fs - fStart[0] + 1 for fs in fStart]

    flds = list(s1.keys())
    dict = dicm_dict(s.Manufacturer, flds)
    len16 = ['AE', 'AS', 'AT', 'CS', 'DA', 'DS', 'DT', 'FD', 'FL', 'IS', 'LO', 'LT', 'PN', 'SH', 'SL', 'SS', 'ST', 'TM', 'UI', 'UL', 'US']
    chDat = ['AE', 'AS', 'CS', 'DA', 'DT', 'LO', 'LT', 'PN', 'SH', 'ST', 'TM', 'UI', 'UT']
    nf = len(iFrame)

    for fld in flds:
        k = [idx for idx, name in enumerate(dict['name']) if name == fld][-1]
        vr = dict['vr'][k]
        group = dict['group'][k]
        isBE = be and group != 2
        isEX = expl or group == 2
        tg = (group << 16 | dict['element'][k]).to_bytes(4, 'big' if isBE else 'little')
        if isEX:
            tg += vr.encode()

        ind = [i for i in range(len(b)) if b[i:i+len(tg)] == tg]
        ind = [i for i in ind if i % 2 > 0]

        if not ind:
            ind = [i for i in range(len(b0)) if b0[i:i+len(tg)] == tg]
            ind = [i for i in ind if i % 2 > 0]
            if ind:
                k = ind[0] + len(tg)
                n, nvr = val_len(vr, b0[k:k+6], isEX, isBE)
                k += nvr
                a = read_val(b0[k:k+n], vr, isBE)
                if isinstance(a, str):
                    a = [a]
                s1[fld] = [a] * nf
            continue

        len_bytes = 4
        if not isEX:
            ind = [i + 4 for i in ind]
        elif vr in len16:
            ind = [i + 6 for i in ind]
            len_bytes = 2
        else:
            ind = [i + 8 for i in ind]

        isCH = vr in chDat
        isDS = vr in ['DS', 'IS']
        if not isCH and not isDS:
            fmt = vr2fmt(vr)
            if not fmt:
                continue

        for k in range(nf):
            j = iFrame[k]
            j = next((i for i in ind if fStart[j] < i < fStart[j+1]), None)
            if j is None:
                continue

            if len_bytes == 2:
                n = int.from_bytes(b[j:j+2], 'big' if isBE else 'little')
            else:
                n = int.from_bytes(b[j:j+4], 'big' if isBE else 'little')

            a = b[j+len_bytes:j+len_bytes+n]
            if isDS:
                a = list(map(float, a.decode().split('\\')))
                try:
                    s1[fld][:, k] = a
                except:
                    pass
            elif isCH:
                a = a.rstrip(b'\x00').decode().strip()
                try:
                    s1[fld][k] = a
                except:
                    pass
            else:
                a = np.frombuffer(a, dtype=fmt)
                if isBE:
                    a.byteswap(inplace=True)
                try:
                    s1[fld][:, k] = a
                except:
                    pass

    return s1

def update_vendor(p, vendor):
    if p.dict.get('vendor') and vendor.lower().startswith(p.dict['vendor'][:2].lower()):
        dict_ = p.dict  # in case dicm_hdr asks 3rd output
        return p, dict_
    
    dict_full = dicm_dict(vendor)
    
    if not p.fullHdr and 'fields' in p.dict:
        dict_ = dicm_dict(vendor, p.dict['fields'])
    else:
        dict_ = dict_full
    
    p.dict = dict_
    return p, dict_

def dicm_hdr(fname, dict=None, iFrames=None):
    if dict is None:
        dict = dicm_dict()

    s = {}
    info = ''
    p = {
        'fullHdr': False,
        'dict': dict,
        'iFrames': iFrames or []
    }
    
    try:
        with open(fname, 'rb') as f:
            f.seek(0, os.SEEK_END)
            fSize = f.tell()
            f.seek(0)
            b8 = f.read(130000)

            if fSize < 140:
                return {}, 'Invalid file: ' + fname

            iTagStart = 132
            isDicm = b8[128:132] == b'DICM'
            if not isDicm:
                group = ch2int16(b8[0:2], False)
                isDicm = group in [2, 8]
                iTagStart = 0
            
            if not isDicm:
                # Handle other file formats like PAR/HEAD
                ext = os.path.splitext(fname)[1].upper()
                if ext == '.PAR':
                    return philips_par(fname)
                elif ext == '.HEAD':
                    return afni_head(fname)
                else:
                    return {}, 'Unknown file type: ' + fname

            p['expl'] = False
            p['be'] = False

            tsUID = ''
            i = b8.find(b'\x02\x00\x10\x00UI')
            if i >= 0:
                i += 6
                n = ch2int16(b8[i:i+2], False)
                tsUID = dcm_str(b8[i+2:i+2+n])
                p['expl'] = tsUID != '1.2.840.10008.1.2'
                p['be'] = tsUID == '1.2.840.10008.1.2.2'

            tg = b'\xE0\x7F\x10\x00'
            if p['be']:
                tg = tg[::-1]
            
            found = False
            for nb in [0, 2000000, 20000000, fSize]:
                b8 += f.read(nb)
                i = b8.find(tg)
                if i >= 0:
                    p['iPixelData'] = i + 7 if p['expl'] else i + 11
                    if len(b8) < p['iPixelData'] + 4:
                        b8 += f.read(8)
                    p['bytes'] = ch2int32(b8[p['iPixelData']-4:p['iPixelData']], p['be'])
                    if p['bytes'] == 4294967295 and f.tell() == fSize:
                        break
                    d = fSize - p['iPixelData'] - p['bytes']
                    if 0 <= d < 16:
                        found = True
                        break
                if f.tell() == fSize:
                    if not i:
                        return {}, 'No PixelData in ' + fname
                    break

            s['Filename'] = fname
            s['FileSize'] = fSize

            # More parsing logic...
    
    except Exception as e:
        return {}, str(e)

    return s, info

def read_par_file(fname):
    def par_key(key, is_num=True):
        pattern = re.compile(r'\.\s*' + re.escape(key) + r'\s*:\s*(.*)')
        match = pattern.search(par_contents)
        if match:
            value = match.group(1).strip()
            if is_num:
                try:
                    return float(value)
                except ValueError:
                    return None
            return value
        return None if is_num else ''

    def get_table_val(key, fldname, i_row=None):
        if i_row is None:
            i_row = np.arange(nImg)
        if key in col_labels:
            col_idx = col_labels.index(key)
            setattr(s, fldname, para_table[i_row, col_start[col_idx]:col_start[col_idx+1]])
    
    if fname.lower().endswith('.rec'):
        fname = fname[:-4] + '.PAR'
        if not os.path.exists(fname):
            fname = fname[:-4] + '.par'
    
    if not os.path.exists(fname):
        return None, f'File not exist: {fname}'
    
    with open(fname, 'r') as f:
        par_contents = f.read()
    
    par_contents = par_contents.replace('\r', '\n').replace('\n\n', '\n')
    while True:
        new_str = par_contents.replace('\n.  ', '\n. ')
        if new_str == par_contents:
            break
        par_contents = new_str
    
    if 'image export tool' not in par_contents.lower():
        return None, 'Not PAR file'
    
    s = {
        'SoftwareVersion': re.search(r'image export tool\s*(.*)', par_contents, re.IGNORECASE).group(1) + '\PAR'
    }
    
    if s['SoftwareVersion'].lower().startswith('v3'):
        print('V3 PAR file is not supported.')
        return None, 'V3 PAR file is not supported.'
    
    s['PatientName'] = par_key('Patient name', is_num=False)
    s['StudyDescription'] = par_key('Examination name', is_num=False)
    s['SeriesDescription'] = os.path.basename(fname)
    s['ProtocolName'] = par_key('Protocol name', is_num=False)
    
    date_time = par_key('Examination date/time', is_num=False)
    s['AcquisitionDateTime'] = re.sub(r'\D', '', date_time)
    
    s['SeriesNumber'] = par_key('Acquisition nr')
    s['SeriesInstanceUID'] = f"{s['SeriesNumber']}.{datetime.now().strftime('%y%m%d.%H%M%S.%f')}{np.random.randint(1e9):09}"
    
    s['NumberOfEchoes'] = par_key('Max. number of echoes')
    nSL = par_key('Max. number of slices/locations')
    s['LocationsInAcquisition'] = nSL
    
    patient_position = par_key('Patient position', is_num=False) or par_key('Patient Position', is_num=False)
    if patient_position:
        s['PatientPosition'] = ''.join(re.findall(r'\b\w', patient_position))
    
    s['MRAcquisitionType'] = par_key('Scan mode', is_num=False)
    s['ScanningSequence'] = par_key('Technique', is_num=False)
    series_type = par_key('Series Type', is_num=False).replace(' ', '')
    s['ImageType'] = f"PhilipsPAR\\{series_type}\\{s['ScanningSequence']}"
    
    s['RepetitionTime'] = par_key('Repetition time')
    s['WaterFatShift'] = par_key('Water Fat shift')
    
    rot_angle = par_key('Angulation midslice')
    if rot_angle:
        rot_angle = np.array(rot_angle.split(), dtype=float)[[2, 0, 1]]
    
    pos_mid = par_key('Off Centre midslice')
    if pos_mid:
        pos_mid = np.array(pos_mid.split(), dtype=float)
        s['Stack'] = {
            'Item_1': {
                'MRStackOffcentreAP': pos_mid[0],
                'MRStackOffcentreFH': pos_mid[1],
                'MRStackOffcentreRL': pos_mid[2]
            }
        }
        pos_mid = pos_mid[[2, 0, 1]]
    
    s['EPIFactor'] = par_key('EPI factor')
    is_DTI = par_key('Diffusion') > 0
    if is_DTI:
        s['ImageType'] += '\\DIFFUSION\\'
        s['DiffusionEchoTime'] = par_key('Diffusion echo time')
    
    preparation_direction = par_key('Preparation direction', is_num=False)
    if preparation_direction:
        preparation_direction = ''.join(re.findall(r'\b\w', preparation_direction))
        s['Stack']['Item_1']['MRStackPreparationDirection'] = preparation_direction
        i_phase = 'LRAPFH'.find(preparation_direction[0]) // 2 + 1
    
    table_start_idx = par_contents.rfind('IMAGE INFORMATION DEFINITION')
    table_end_idx = par_contents.find('= IMAGE INFORMATION =', table_start_idx)
    column_def = par_contents[table_start_idx:table_end_idx]
    col_labels, col_start = [], []
    for line in column_def.split('\n'):
        if line.strip().startswith('#'):
            parts = line.strip().split()
            if parts[-1].startswith('<'):
                col_labels.append(parts[1])
                col_start.append(int(parts[-1][1:-1]))
    col_start = np.cumsum([1] + col_start)
    
    image_info_start = par_contents.find('= IMAGE INFORMATION =', table_end_idx) + len('= IMAGE INFORMATION =')
    image_data = par_contents[image_info_start:]
    image_data_lines = [line for line in image_data.split('\n') if line.strip()]
    
    first_row = np.fromstring(image_data_lines[0], sep=' ')
    n_items = len(first_row)
    all_data = np.fromstring(' '.join(image_data_lines), sep=' ').reshape(-1, n_items)
    
    nImg = all_data.shape[0]
    para_table = all_data[:nImg]
    
    s['NumberOfFrames'] = nImg
    nVol = nImg // nSL
    s['NumberOfTemporalPositions'] = nVol
    
    dim3_is_volume = np.all(np.diff(para_table[:, col_start[col_labels.index('slice number')]]) == 0)
    s['Dim3IsVolume'] = dim3_is_volume
    if dim3_is_volume:
        iVol = np.arange(nVol)
        iSL = np.arange(0, nImg, nVol)
    else:
        iVol = np.arange(0, nImg, nSL)
        iSL = np.arange(nSL)
    
    slice_numbers = para_table[iSL, col_start[col_labels.index('slice number')]]
    if np.any(np.diff(slice_numbers, 2) > 0):
        s['SliceNumber'] = slice_numbers
    
    image_type_mr = para_table[iVol, col_start[col_labels.index('image_type_mr')]]
    if np.any(np.diff(image_type_mr) != 0):
        s['ComplexImageComponent'] = 'MIXED'
        s['VolumeIsPhase'] = (image_type_mr == 3)
        s['LastFile'] = {
            'RescaleIntercept': para_table[-1, col_start[col_labels.index('rescale intercept')]],
            'RescaleSlope': para_table[-1, col_start[col_labels.index('rescale slope')]]
        }
    else:
        s['ComplexImageComponent'] = 'MAGNITUDE' if image_type_mr[0] == 0 else 'PHASE'
    
    common_cols = [
        'image pixel size', 'recon resolution', 'image angulation',
        'slice thickness', 'slice gap', 'slice orientation', 'pixel spacing'
    ]
    if s['ComplexImageComponent'] != 'MIXED':
        common_cols += ['rescale intercept', 'rescale slope']
    
    for col in common_cols:
        if col in col_labels:
            col_idx = col_labels.index(col)
            if not np.allclose(np.diff(para_table[:, col_start[col_idx]:col_start[col_idx+1]]), 0):
                return None, f'Inconsistent image size, bits etc: {fname}'
    
    get_table_val('image pixel size', 'BitsAllocated')
    get_table_val('recon resolution', 'Columns')
    s['Rows'] = s['Columns'][1]
    s['Columns'] = s['Columns'][0]
    get_table_val('rescale intercept', 'RescaleIntercept')
    get_table_val('rescale slope', 'RescaleSlope')
    get_table_val('window center', 'WindowCenter')
    get_table_val('window width', 'WindowWidth')
    
    max_wc = np.max(s['WindowCenter'] + s['WindowWidth'] / 2)
    min_wc = np.min(s['WindowCenter'] - s['WindowWidth'] / 2)
    s['WindowCenter'] = round((max_wc + min_wc) / 2)
    s['WindowWidth'] = np.ceil(max_wc - min_wc)
    
    get_table_val('slice thickness', 'SliceThickness')
    get_table_val('echo_time', 'EchoTime')
    get_table_val('image_flip_angle', 'FlipAngle')
    get_table_val('number of averages', 'NumberOfAverages')
    
    if is_DTI:
        get_table_val('diffusion_b_factor', 'B_value', iVol)
        get_table_val('diffusion', 'bvec_original', iVol)
        if 'bvec_original' in s:
            s['bvec_original'] = s['bvec_original'][:, [2, 0, 1]]
    
    get_table_val('TURBO factor', 'TurboFactor')
    
    # Calculate rotation matrix
    if rot_angle:
        ca, sa = np.cos(np.radians(rot_angle)), np.sin(np.radians(rot_angle))
        rx = np.array([[1, 0, 0], [0, ca[0], -sa[0]], [0, sa[0], ca[0]]])
        ry = np.array([[ca[1], 0, sa[1]], [0, 1, 0], [-sa[1], 0, ca[1]]])
        rz = np.array([[ca[2], -sa[2], 0], [sa[2], ca[2], 0], [0, 0, 1]])
        R = np.dot(np.dot(rx, ry), rz)
    
    get_table_val('slice orientation', 'SliceOrientation')
    iOri = (s['SliceOrientation'] + 1) % 3 + 1
    if iOri == 1:
        s['SliceOrientation'] = 'SAGITTAL'
        R[:, [0, 2]] = -R[:, [0, 2]]
        R = R[:, [1, 2, 0]]
    elif iOri == 2:
        s['SliceOrientation'] = 'CORONAL'
        R[:, 2] = -R[:, 2]
        R = R[:, [0, 2, 1]]
    else:
        s['SliceOrientation'] = 'TRANSVERSAL'
    
    get_table_val('pixel spacing', 'PixelSpacing')
    s['PixelSpacing'] = s['PixelSpacing'].flatten()
    get_table_val('slice gap', 'SpacingBetweenSlices')
    s['SpacingBetweenSlices'] = s['SpacingBetweenSlices'] + s['SliceThickness']
    
    if 'iPhase' in locals():
        phase_dir = 'COL' if i_phase == (iOri == 1) + 1 else 'ROW'
        s['InPlanePhaseEncodingDirection'] = phase_dir
    
    s['ImageOrientationPatient'] = R[:6].flatten()
    R = np.dot(R, np.diag([*s['PixelSpacing'], s['SpacingBetweenSlices']]))
    R = np.vstack([R, pos_mid, [0, 0, 0, 1]])
    
    get_table_val('image offcentre', 'SliceLocation')
    s['SliceLocation'] = s['SliceLocation'][iOri]
    if np.sign(R[iOri, 2]) != np.sign(pos_mid[iOri] - s['SliceLocation']):
        R[:, 2] = -R[:, 2]
    
    R[:, 3] = np.dot(R, [-((s['Columns'] - 1) / 2), -((s['Rows'] - 1) / 2), -((nSL - 1) / 2), 1])
    y = np.dot(R, [0, 0, nSL - 1, 1])
    s['ImagePositionPatient'] = R[:3, 3]
    s['LastFile']['ImagePositionPatient'] = y[:3]
    
    s['Manufacturer'] = 'Philips'
    s['Filename'] = os.path.join(os.path.dirname(fname), os.path.basename(fname).replace('.PAR', '.REC'))
    s['PixelData'] = {
        'Start': 0,
        'Bytes': int(s['Rows'] * s['Columns'] * nImg * s['BitsAllocated'] / 8)
    }
    
    return s, ''

SN = 1

def afni_head(fname):
    global SN
    err = ''
    s = {}

    if len(fname) > 5 and fname[-5:] == '.BRIK':
        fname = fname[:-5] + '.HEAD'

    if not os.path.isfile(fname):
        return {}, f'File not exist: {fname}'

    with open(fname, 'r') as file:
        str_content = file.read()

    i = str_content.find('DATASET_DIMENSIONS')
    if i == -1:
        return {}, 'Not brik header file'

    _, foo = os.path.split(fname)
    s['ProtocolName'] = foo
    s['SeriesNumber'] = SN
    SN += 1
    s['SeriesInstanceUID'] = f"{s['SeriesNumber']}.{datetime.now():%y%m%d.%H%M%S.%f}{random.randint(0, 1e9):09}"
    s['ImageType'] = f"AFNIHEAD\\{afni_key('TYPESTRING', str_content)}"

    foo = afni_key('BYTEORDER_STRING', str_content)
    if foo.startswith('M'):
        return {}, 'BYTEORDER_STRING not supported'

    foo = afni_key('BRICK_FLOAT_FACS', str_content)
    if np.any(np.diff(foo) != 0):
        return {}, 'Inconsistent BRICK_FLOAT_FACS'

    if foo[0] == 0:
        foo = [1]
    s['RescaleSlope'] = foo[0]
    s['RescaleIntercept'] = 0

    foo = afni_key('BRICK_TYPES', str_content)
    if np.any(np.diff(foo) != 0):
        return {}, 'Inconsistent DataType'

    foo = foo[0]
    if foo == 0:
        s['BitsAllocated'] = 8
        s['PixelData'] = {'Format': '*uint8'}
    elif foo == 1:
        s['BitsAllocated'] = 16
        s['PixelData'] = {'Format': '*int16'}
    elif foo == 3:
        s['BitsAllocated'] = 32
        s['PixelData'] = {'Format': '*single'}
    else:
        raise ValueError(f'Unsupported BRICK_TYPES: {foo}')

    hist = afni_key('HISTORY_NOTE', str_content)
    i = hist.find('Time:') + 6
    if i >= 6:
        dat = datetime.strptime(hist[i:i+11], '%b %d %Y')
        s['AcquisitionDateTime'] = dat.strftime('%Y%m%d')
    i = hist.find('Sequence:') + 9
    if i >= 9:
        s['ScanningSequence'] = hist[i:].split(' ')[0]
    i = hist.find('Studyid:') + 8
    if i >= 8:
        s['StudyID'] = hist[i:].split(' ')[0]
    i = hist.find('TE:') + 3
    if i >= 3:
        s['EchoTime'] = float(hist[i:].split(' ')[0]) * 1000

    foo = afni_key('SCENE_DATA', str_content)
    s['TemplateSpace'] = foo[0] + 1
    if foo[1] == 9:
        s['ImageType'] += '\\DIFFUSION\\'

    dim = afni_key('DATASET_DIMENSIONS', str_content)
    s['Columns'] = dim[0]
    s['Rows'] = dim[1]
    s['LocationsInAcquisition'] = dim[2]

    R = afni_key('IJK_TO_DICOM_REAL', str_content)
    if not R:
        R = afni_key('IJK_TO_DICOM', str_content)
    R = np.reshape(R, (4, 3)).T
    s['ImagePositionPatient'] = R[:, 3]
    y = np.dot(np.vstack([R, [0, 0, 0, 1]]), [0, 0, dim[2] - 1, 1])
    s['LastFile'] = {'ImagePositionPatient': y[:3]}
    R = R[:3, :3]
    R = R / np.sqrt(np.sum(R ** 2, axis=0))
    s['ImageOrientationPatient'] = R[:3, :2].flatten()
    foo = afni_key('DELTA', str_content)
    s['PixelSpacing'] = foo[:2]
    s['SliceThickness'] = foo[2]
    foo = afni_key('BRICK_STATS', str_content)
    foo = np.reshape(foo, (2, len(foo) // 2))
    mn = np.min(foo[0, :])
    mx = np.max(foo[1, :])
    s['WindowCenter'] = (mx + mn) / 2
    s['WindowWidth'] = mx - mn
    foo = afni_key('TAXIS_FLOATS', str_content)
    if foo:
        s['RepetitionTime'] = foo[1] * 1000
    foo = afni_key('TAXIS_NUMS', str_content)
    if foo:
        inMS = foo[2] == 77001
        foo = afni_key('TAXIS_OFFSETS', str_content)
        if inMS:
            foo = foo / 1000
        if foo:
            s['MosaicRefAcqTimes'] = foo

    foo = afni_key('DATASET_RANK', str_content)
    dim.append(foo[1])
    s['NumberOfTemporalPositions'] = dim[4]

    s['Manufacturer'] = ''
    s['Filename'] = fname.replace('.HEAD', '.BRIK')
    s['PixelData']['Start'] = 0
    s['PixelData']['Bytes'] = np.prod(dim[:4]) * s['BitsAllocated'] // 8

    return s, err

def afni_key(key, str_content):
    key_line = f'\nname = {key}\n'
    i1 = str_content.find(key_line)
    if i1 == -1:
        return []
    i1 += len(key_line)
    i2 = str_content[:i1].rfind('\ntype = ')
    key_type = str_content[i2:i1].split('=')[-1].strip().split('-')[0]
    i1 = str_content[i1:].find('\n') + i1
    count = int(str_content[i1:].split('=')[-1].split()[0])
    if key_type == 'string':
        i1 = str_content[i1:].find("'") + i1 + 1
        val = str_content[i1:i1 + count - 1]
    else:
        i1 = str_content[i1:].find('\n') + i1 + 1
        val = list(map(float, str_content[i1:].split()[:count]))
    return val

def bv_file(fname):
    s = {}
    err = ''

    try:
        bv = BVQXfile(fname)  # Assuming BVQXfile is a class from a hypothetical bvqxtools package
    except Exception as e:
        err = str(e)
        if 'UndefinedFunction' in str(e):
            print('Please download BVQXtools at \nhttp://support.brainvoyager.com/available-tools/52-matlab-tools-bvxqtools.html', file=sys.stderr)
        return s, err

    if bv.Trf:
        for trf in bv.Trf:
            if not np.array_equal(np.diag(trf.TransformationValues), [1, 1, 1, 1]):
                err = 'Data has been transformed: skipped.'
                return s, err

    global SN, subj, folder
    if 'SN' not in globals():
        SN = 1
        subj = ''
        folder = ''

    s['Filename'] = bv.FilenameOnDisk
    fType = bv.filetype
    s['ImageType'] = f'BrainVoyagerFile\\{fType}'

    pth, nam = os.path.split(s['Filename'])
    s['SeriesDescription'] = nam
    if folder == '' or folder != pth:
        folder = pth
        subj = ''
        if fType in ['fmr', 'dmr']:
            _, nam = os.path.split(bv.FirstDataSourceFile)
            nam = nam.split('-')[0]
            if nam:
                subj = nam
        else:
            fnames = [f for f in os.listdir(pth) if f.endswith('.fmr')]
            if not fnames:
                fnames = [f for f in os.listdir(pth) if f.endswith('.dmr')]
            if fnames:
                bv1 = BVQXfile(os.path.join(pth, fnames[0]))
                _, nam = os.path.split(bv1.FirstDataSourceFile)
                bv1.ClearObject()
                nam = nam.split('-')[0]
                if nam:
                    subj = nam

    if subj:
        s['PatientName'] = subj

    s['SoftwareVersion'] = f'{bv.FileVersion}/BV_FileVersion'
    s['Columns'] = bv.NCols
    s['Rows'] = bv.NRows
    s['SliceThickness'] = bv.SliceThickness
    R = np.array([[bv.RowDirX, bv.RowDirY, bv.RowDirZ],
                  [bv.ColDirX, bv.ColDirY, bv.ColDirZ]]).T
    s['ImageOrientationPatient'] = R.flatten()
    R = np.column_stack((R, np.cross(R[:, 0], R[:, 1])))
    ixyz = np.argmax(np.abs(R), axis=0)
    iSL = ixyz[2]

    try:
        s['TemplateSpace'] = bv.ReferenceSpace
        if s['TemplateSpace'] == 0:
            s['TemplateSpace'] = 1
    except AttributeError:
        s['TemplateSpace'] = 1

    pos = np.array([[bv.Slice1CenterX, bv.Slice1CenterY, bv.Slice1CenterZ],
                    [bv.SliceNCenterX, bv.SliceNCenterY, bv.SliceNCenterZ]]).T

    if fType == 'vmr':
        s['SpacingBetweenSlices'] = s['SliceThickness'] + bv.GapThickness
        s['PixelSpacing'] = np.array([bv.VoxResX, bv.VoxResY])
        if bv.VMRData16 is not None:
            nSL = bv.DimZ
            s['PixelData'] = bv.VMRData16
        else:
            v16 = s['Filename'][:-3] + 'v16'
            if os.path.isfile(v16):
                bv16 = BVQXfile(v16)
                nSL = bv16.DimZ
                s['PixelData'] = bv16.VMRData
                bv16.ClearObject()
            else:
                ix = (bv.DimX - s['Columns']) // 2
                iy = (bv.DimY - s['Rows']) // 2
                R3 = abs(R[iSL, 2]) * s['SpacingBetweenSlices']
                nSL = round(abs(np.diff(pos[iSL, :])) / R3) + 1
                iz = (bv.DimZ - nSL) // 2
                s['PixelData'] = bv.VMRData[ix:ix + s['Columns'], iy:iy + s['Rows'], iz:iz + nSL, :]

        s['LocationsInAcquisition'] = nSL
        s['MRAcquisitionType'] = '3D'

    elif fType in ['fmr', 'dmr']:
        s['SpacingBetweenSlices'] = s['SliceThickness'] + bv.SliceGap
        s['PixelSpacing'] = np.array([bv.InplaneResolutionX, bv.InplaneResolutionY])
        nSL = bv.NrOfSlices
        s['LocationsInAcquisition'] = nSL
        s['NumberOfTemporalPositions'] = bv.NrOfVolumes
        s['RepetitionTime'] = bv.TR
        s['EchoTime'] = bv.TE
        if bv.TimeResolutionVerified:
            order_map = {
                1: list(range(1, nSL + 1)),
                2: list(range(nSL, 0, -1)),
                3: list(range(1, nSL + 1, 2)) + list(range(2, nSL + 1, 2)),
                4: list(range(nSL, 0, -2)) + list(range(nSL - 1, 0, -2)),
                5: list(range(2, nSL + 1, 2)) + list(range(1, nSL + 1, 2)),
                6: list(range(nSL - 1, 0, -2)) + list(range(nSL, 0, -2))
            }
            ind = order_map.get(bv.SliceAcquisitionOrder, [])
            if ind:
                t = np.arange(s['LocationsInAcquisition']) * bv.InterSliceTime
                t = t[ind]
                s['SliceTiming'] = t

        if fType == 'fmr':
            bv.LoadSTC()
            s['PixelData'] = np.transpose(bv.Slice[0].STCData, (0, 1, 3, 2))
            for i in range(1, len(bv.Slice)):
                s['PixelData'][:, :, i, :] = np.transpose(bv.Slice[i].STCData, (0, 1, 3, 2))
        else:
            s['ImageType'] = f'{s["ImageType"]}\\DIFFUSION\\'
            bv.LoadDWI()
            s['PixelData'] = bv.DWIData
            if bv.GradientInformationAvailable.startswith('Y'):
                a = bv.GradientInformation
                s['B_value'] = a[:, 3]
                a = a[:, :3]
                s['bvec_original'] = a

        if np.issubdtype(s['PixelData'].dtype, np.integer) and \
                s['PixelData'].max() < 32768 and s['PixelData'].min() >= -32768:
            s['PixelData'] = s['PixelData'].astype(np.int16)

    else:
        err = f'Unknown BV file type: {fType}'
        s = {}
        return s, err

    pos = pos - R[:, :2] @ np.diag(s['PixelSpacing']) @ np.array([s['Columns'], s['Rows']]) / 2
    s['ImagePositionPatient'] = pos[:, 0]
    s['LastFile.ImagePositionPatient'] = pos[:, 1]

    try:
        _, nam = os.path.split(bv.FirstDataSourceFile)
        serN = float(nam.split('-')[1])
        if serN:
            SN = serN
    except:
        pass

    s['SeriesNumber'] = SN
    SN += 1
    s['SeriesInstanceUID'] = f'{s["SeriesNumber"]}.{datetime.datetime.now().strftime("%y%m%d.%H%M%S.%f")}.{random.randint(0, 1e9):09.0f}'
    c = s['PixelData'].dtype.name
    if c == 'float64':
        s['BitsAllocated'] = 64
    elif c == 'float32':
        s['BitsAllocated'] = 32
    else:
        s['BitsAllocated'] = int(''.join(filter(str.isdigit, c)))

    return s, err

