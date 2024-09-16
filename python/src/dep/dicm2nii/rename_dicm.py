import os
import re
from pydicom import dcmread
from tkinter import filedialog
from tkinter import Tk

def rename_dicm(files=None, fmt=None):
    if files is None:
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory(initialdir=os.getcwd(), title="Select a folder containing DICOM files")
        if not folder:
            return
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        options = '''Choose Output format: 
1: run1_00001.dcm (SeriesDescription_instance)
2: BrainVoyager format (subj-series-acquisition-instance)
3: run1_001_00001.dcm (SeriesDescription_series_instance)
4: subj_run1_00001.dcm (subj_SeriesDescription_instance)
5: run1_001_001_00001.dcm (SeriesDescription_series_acquisition_instance)'''

        fmt = input(f'{options}\nRename Dicom: ')
        if not fmt:
            return
        fmt = int(fmt)
    else:
        if os.path.isdir(files):
            folder = files
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        else:
            if not isinstance(files, list):
                files = [files]
            folder = os.path.dirname(files[0])
            if not folder:
                folder = os.getcwd()
        if fmt is None:
            fmt = 5

    nFile = len(files)
    if nFile < 1:
        return
    if not folder.endswith(os.sep):
        folder += os.sep

    err = ''
    print(f' Renaming DICOM files: 1/{nFile}', end='')

    for i, file in enumerate(files, start=1):
        print(f'\r Renaming DICOM files: {i}/{nFile}', end='')
        try:
            s = dcmread(os.path.join(folder, file))
            sN = getattr(s, 'SeriesNumber', None)
            aN = getattr(s, 'AcquisitionNumber', None)
            iN = getattr(s, 'InstanceNumber', None)
            manufacturer = getattr(s, 'Manufacturer', '').strip()
            pName = (getattr(s, 'ProtocolName', '') if manufacturer.startswith('SIEMENS') 
                     else getattr(s, 'SeriesDescription', '')).strip()
            sName = getattr(s, 'PatientName', getattr(s, 'PatientID', '')).strip()
        except Exception as e:
            continue

        pName = re.sub(r'[^a-zA-Z0-9]', '_', pName)
        sName = re.sub(r'[^a-zA-Z0-9]', '_', sName)

        if manufacturer.startswith('Philips'):
            sN = aN
        elif manufacturer.startswith('SIEMENS') and hasattr(s, 'EchoNumber') and s.EchoNumber > 1:
            aN = s.EchoNumber

        if fmt == 1:
            name = f'{pName}_{iN:05d}.dcm'
        elif fmt == 2:
            name = f'{sName}-{sN:04d}-{aN:04d}-{iN:05d}.dcm'
        elif fmt == 3:
            name = f'{pName}_{s.SeriesNumber:02d}_{iN:05d}.dcm'
        elif fmt == 4:
            name = f'{sName}_{pName}_{iN:05d}.dcm'
        elif fmt == 5:
            name = f'{pName}_{sN:03d}_{aN:03d}_{iN:05d}.dcm'
        else:
            raise ValueError('Invalid format.')

        if file == name:
            continue

        try:
            os.rename(os.path.join(folder, file), os.path.join(folder, name))
        except Exception as e:
            err += f'{file}: {str(e)}\n'

    print('\n')
    if err:
        print(f'Errors encountered:\n{err}', file=sys.stderr)

