import os
import gzip
import shutil
import tempfile
import numpy as np

def load_nii_ext(filename):
    if not filename:
        raise ValueError('Usage: ext = load_nii_ext(filename)')

    v = np.__version__

    # Check file extension. If .gz, unpack it into temp folder
    if filename.endswith('.gz'):
        if not (filename.endswith('.img.gz') or filename.endswith('.hdr.gz') or filename.endswith('.nii.gz')):
            raise ValueError('Please check filename.')

        if float(v[:3]) < 1.20:
            raise ValueError('Please use numpy 1.20 and above, or run gunzip outside Python.')

        tmp_dir = tempfile.mkdtemp()

        if filename.endswith('.img.gz'):
            filename1 = filename
            filename2 = filename[:-7] + '.hdr.gz'
            filename1 = gunzip_file(filename1, tmp_dir)
            filename2 = gunzip_file(filename2, tmp_dir)
            filename = filename1
        elif filename.endswith('.hdr.gz'):
            filename1 = filename
            filename2 = filename[:-7] + '.img.gz'
            filename1 = gunzip_file(filename1, tmp_dir)
            filename2 = gunzip_file(filename2, tmp_dir)
            filename = filename1
        elif filename.endswith('.nii.gz'):
            filename = gunzip_file(filename, tmp_dir)

    machine = '<'
    new_ext = 0

    if filename.endswith('.nii'):
        new_ext = 1
        filename = filename[:-4]

    if filename.endswith('.hdr'):
        filename = filename[:-4]

    if filename.endswith('.img'):
        filename = filename[:-4]

    if new_ext:
        fn = f'{filename}.nii'
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{filename}.nii".')
    else:
        fn = f'{filename}.hdr'
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Cannot find file "{filename}.hdr".')

    try:
        with open(fn, 'rb') as fid:
            vox_offset = 0
            fid.seek(0, os.SEEK_SET)
            if np.fromfile(fid, dtype=np.int32, count=1)[0] == 348:
                if new_ext:
                    fid.seek(108, os.SEEK_SET)
                    vox_offset = np.fromfile(fid, dtype=np.float32, count=1)[0]

                ext = read_extension(fid, vox_offset)
            else:
                # first try reading the opposite endian to 'machine'
                machine = '>' if machine == '<' else '<'
                fid = open(fn, 'rb')
                fid.seek(0, os.SEEK_SET)

                if np.fromfile(fid, dtype=np.int32, count=1)[0] != 348:
                    raise ValueError(f'File "{fn}" is corrupted.')

                if new_ext:
                    fid.seek(108, os.SEEK_SET)
                    vox_offset = np.fromfile(fid, dtype=np.float32, count=1)[0]

                ext = read_extension(fid, vox_offset)
    finally:
        if 'tmp_dir' in locals():
            shutil.rmtree(tmp_dir)

    return ext


def gunzip_file(filename, tmp_dir):
    with gzip.open(filename, 'rb') as f_in:
        with open(os.path.join(tmp_dir, os.path.basename(filename)[:-3]), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return os.path.join(tmp_dir, os.path.basename(filename)[:-3])


def read_extension(fid, vox_offset):
    ext = {}

    if vox_offset:
        end_of_ext = vox_offset
    else:
        fid.seek(0, os.SEEK_END)
        end_of_ext = fid.tell()

    if end_of_ext > 352:
        fid.seek(348, os.SEEK_SET)
        ext['extension'] = np.fromfile(fid, dtype=np.uint8, count=4)

    if not ext or ext['extension'][0] == 0:
        return {}

    i = 1
    ext['section'] = []

    while fid.tell() < end_of_ext:
        section = {}
        section['esize'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
        section['ecode'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
        section['edata'] = fid.read(section['esize'] - 8).decode('utf-8')
        ext['section'].append(section)
        i += 1

    ext['num_ext'] = len(ext['section'])

    return ext
