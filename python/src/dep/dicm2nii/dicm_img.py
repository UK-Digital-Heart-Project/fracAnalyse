import pydicom
import numpy as np
from PIL import Image
import os
import gzip
import shutil

def dicm_img(s, xpose=True):
    # Persistent variables in Python
    if not hasattr(dicm_img, "flds"):
        dicm_img.flds = ['Columns', 'Rows', 'BitsAllocated']
    if not hasattr(dicm_img, "dict"):
        dicm_img.dict = None

    def dicm_dict(*args):
        # Simulate dicm_dict function in Python
        return {field: None for field in args}
    
    def dicm_hdr(filename, dict):
        # Simulate dicm_hdr function in Python using pydicom
        try:
            ds = pydicom.dcmread(filename)
            hdr = {elem.tag: elem.value for elem in ds.iterall()}
            return hdr, None
        except Exception as e:
            return None, str(e)
    
    if isinstance(s, dict) and not all(key in s for key in dicm_img.flds + ['PixelData']):
        s = s['Filename']
    
    if isinstance(s, str):
        if dicm_img.dict is None:
            dicm_img.dict = dicm_dict(*dicm_img.flds, 'SamplesPerPixel', 'PixelRepresentation',
                                      'PlanarConfiguration', 'BitsStored', 'HighBit')
        s, err = dicm_hdr(s, dicm_img.dict)
        if s is None:
            raise ValueError(err)
    
    spp = s.get('SamplesPerPixel', 1)
    
    if isinstance(s['PixelData'], np.ndarray):
        img = s['PixelData']
        return img
    
    fmt = f"uint{s['BitsAllocated']}"
    
    if not xpose:
        xpose = True

    filename = s['Filename']
    if not os.path.exists(filename):
        if os.path.exists(filename + '.gz'):
            with gzip.open(filename + '.gz', 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not exists: {filename}")
    
    with open(filename, 'rb') as fid:
        fid.seek(s['PixelData']['Start'])
        
        if not s.get('TransferSyntaxUID') or \
           s['TransferSyntaxUID'] in ['1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2', '1.2.840.10008.1.2']:
            n = s['PixelData']['Bytes'] // (s['BitsAllocated'] // 8)
            img = np.fromfile(fid, dtype=fmt, count=n)
            
            if all(k in s for k in ['BitsStored', 'HighBit']) and \
               s['BitsStored'] != s['HighBit'] + 1 and s['BitsStored'] != s['BitsAllocated']:
                img = np.left_shift(img, s['BitsStored'] - s['HighBit'] - 1)
            
            dim = [s['Columns'], s['Rows']]
            nFrame = n // (spp * dim[0] * dim[1])
            if not s.get('PlanarConfiguration') or s['PlanarConfiguration'] == 0:
                img = img.reshape([spp, *dim, nFrame])
                img = np.transpose(img, [1, 2, 0, 3])
            else:
                img = img.reshape([dim[0], dim[1], spp, nFrame])
            
            if xpose:
                img = np.transpose(img, [1, 0, 2, 3])
            
            if s.get('TransferSyntaxUID') == '1.2.840.10008.1.2.2':
                img = img.byteswap().newbyteorder()
        else:
            # Handle compressed images
            b = fid.read()
            nEnd = len(b) - 8
            n = int.from_bytes(b[4:8], byteorder='little')
            i = 8 + n
            if n > 0:
                nFrame = n // 4
            else:
                ind = b.find(b'\xFE\xFF\xE0\x00')
                nFrame = len(ind) - 1
            
            img = np.zeros((s['Rows'], s['Columns'], spp, nFrame), dtype=fmt)
            
            fname = 'tempfile'
            for j in range(nFrame):
                i += 4
                n = int.from_bytes(b[i:i+4], byteorder='little')
                i += 4
                with open(fname, 'wb') as temp_f:
                    temp_f.write(b[i:i+n])
                i += n
                img[..., j] = np.array(Image.open(fname))
                if i > nEnd:
                    img[..., j+1:] = []
                    break
            os.remove(fname)
            
            if not xpose:
                img = np.transpose(img, [1, 0, 2, 3])
    
    if s.get('PixelRepresentation') and s['PixelRepresentation'] > 0:
        img = img.astype(fmt.replace('u', ''))
    
    return img
