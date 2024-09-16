import os
import pydicom
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def loadDicom(filepath, batchmode):
    if batchmode == 0:
        I, voxelSize = load_dicom_file(filepath)
    elif batchmode == 1:
        I = {}
        voxelSize = {}
        folderlist = [f for f in os.listdir(filepath) if not f.startswith('.')]
        for folder in folderlist:
            folder_name = f"I{folder}".replace('-', '_')
            I[folder_name], voxelSize[folder_name] = load_dicom_file(os.path.join(filepath, folder))
    return I, voxelSize

def load_dicom_file(filepath):
    dicomlist = [os.path.join(root, file) for root, _, files in os.walk(filepath) for file in files if file.endswith('.dcm')]
    dicomlist = [d for d in dicomlist if not os.path.basename(d).startswith('.')]
    
    if len(dicomlist) == 0:
        return None, None
    
    data_sample = pydicom.dcmread(dicomlist[0])
    mframe = len(data_sample.pixel_array.shape) > 2
    
    if mframe:
        data = data_sample.pixel_array
        dimY, dimX = data.shape[:2]
        numFiles = len(data)
        voxelSize = data_sample.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        
        numPhases = 30
        numSlices = numFiles // numPhases
        
        I = np.zeros((dimY, dimX, numSlices, numPhases), dtype=data.dtype)
        
        tempCnt = 0
        for z in range(numSlices):
            for t in range(numPhases):
                I[:, :, z, t] = data[:, :, tempCnt]
                tempCnt += 1
    else:
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(read_dicom, dicomlist))
        
        data.sort(key=lambda x: (-x['sliceLoc'], x['time']))
        
        dimY, dimX = data[0]['imgData'].shape
        voxelSize = data[0]['pixelSpacing']
        
        numSlices = len(set(round(d['sliceLoc'], 3) for d in data))
        numPhases = len(data) // numSlices
        
        I = np.zeros((dimY, dimX, numSlices, numPhases), dtype=data[0]['imgData'].dtype)
        
        tempCnt = 0
        for z in range(numSlices):
            for t in range(numPhases):
                I[:, :, z, t] = data[tempCnt]['imgData']
                tempCnt += 1
        
        if len(dicomlist) % numSlices != 0:
            print('Dicom Number Error')
    
    return I, voxelSize

def read_dicom(dicom_file):
    imgData = pydicom.dcmread(dicom_file).pixel_array
    info = pydicom.dcmread(dicom_file)
    return {
        'imgData': imgData,
        'sliceLoc': info.SliceLocation,
        'time': info.TriggerTime,
        'pixelSpacing': info.PixelSpacing
    }
