% loadDicom Loads dicoms from parent folder xx/ResearchID/SeriesID
% Copyright (c) 2016 Cai Jiashen

function [I, voxelSize] = loadDicom(filepath,batchmode)

if (batchmode == 0)
    
    [I, voxelSize] = loadDicomFile(filepath);
    
elseif (batchmode == 1)
    
    folderlist = dir(filepath);
    folderlist = folderlist(arrayfun(@(x) ~strcmp(x.name(1),'.'),folderlist));    
    
    for foldercnt = 1:numel(folderlist)
        strcurrfolder = ['I' folderlist(foldercnt).name];
        strcurrfolder = strrep(strcurrfolder,'-','_');
        [I.(strcurrfolder),voxelSize.(strcurrfolder)]=loadDicomFile(fullfile(filepath,folderlist(foldercnt).name));
    end
end

    function [I, voxelSize] = loadDicomFile(filepath)
        
        mframe=0;
        
       % dicomlist = rdir(fullfile(filepath,'**/*'));%.dcm
       % PFT - 22/11/2016
       dicomlist = rdir(fullfile(filepath,'**',filesep,'*.dcm'));%.dcm
        
        dicomlist = dicomlist(~[dicomlist.isdir]);
        dicomlist = dicomlist(arrayfun(@(x) ~strcmp(x.name(1),'.'),dicomlist));
                
        if numel(dicomlist) > 0
            [~, ~, tempext] = fileparts(dicomlist(1).name);
            if strcmpi(tempext,'.DS_STORE')  
                dicomlist(1)=[];
                disp('removed .DS_STORE');
            end
        end
                
        if(numel(dicomlist)==1)
            mframe=1;
        end
        
        %% Multi-frame
        if(mframe==1)
            data=dicm_img(fullfile(dicomlist.name));
            size(data)
            [dimY, dimX, ~, ~]=size(data);
            numFiles = int16(length(data));

            info = dicm_hdr(fullfile(dicomlist.name));            
                        
            disp([dimY, dimX]);
             
            voxelSize=[info.PerFrameFunctionalGroupsSequence.Item_1.PixelMeasuresSequence.Item_1.PixelSpacing(1), info.PerFrameFunctionalGroupsSequence.Item_1.PixelMeasuresSequence.Item_1.PixelSpacing(2)];
            
            disp(voxelSize);
            
            if (isfield(info, 'Private_2001_1017'))         % Philips private tag
                numPhases = int16(info.Private_2001_1017(1));
                numSlices = int16(info.Private_2001_1018(1));
            else
                numPhases = int16(30); % Assume 30 phases
                numSlices = int16(numFiles/numPhases);
            end
            
            I = zeros(dimY,dimX,numSlices,numPhases,class(data));
            
            tempCnt = 1;
            
            for z = 1:numSlices
                for t = 1:numPhases
                    
                    I(:,:,z,t) = data(:,:,:,tempCnt);
                    tempCnt = tempCnt + 1;
                end
            end
        end
        
        
        %% Single Frame
        
        
        if(mframe==0)
            numFiles = numel(dicomlist);
            
            % Check Image Dimensions
            sampInfo = dicm_hdr(fullfile(dicomlist(1).name));
            
            [dimY, dimX] = size(dicm_img(fullfile(dicomlist(1).name)));
            
            disp([dimY, dimX]);
            
            voxelSize=[sampInfo.PixelSpacing(1), sampInfo.PixelSpacing(2)];
            
            disp(voxelSize);
            
            
            element = struct('imgData',zeros(dimY,dimX),'sliceLoc',{[]},'time',{[]});
            data(numel(dicomlist)) = element;
           
            parfor cnt = 1:numel(dicomlist)
                data(cnt).imgData = dicm_img(fullfile(dicomlist(cnt).name));
                
                info = dicm_hdr(fullfile(dicomlist(cnt).name));
                data(cnt).sliceLoc = info.SliceLocation;
                data(cnt).time = info.TriggerTime;
                data(cnt).pixelSpacing = info.PixelSpacing;
                data(cnt).info = info;
            end
            
            % Sort by Slice Location & Phase
            [~, ind] = sortrows([{data.sliceLoc}',{data.time}'], [-1 2]);             
            
            data = data(ind);
            
            numSlices = int16(numel(unique(round([data.sliceLoc],3,'significant'))));
            numPhases = int16(numFiles/numSlices);
            
            imgDataKind = class(data(1).imgData);
            
            I = zeros(dimY,dimX,numSlices,numPhases,imgDataKind);
            
            tempCnt = 1;

            numel(dicomlist);
            
            for z = 1:numSlices
                for t = 1:numPhases
                    I(:,:,z,t) = data(tempCnt).imgData;
                    %I(:,:,z,t) = data(tempCnt).imgData;
                    tempCnt = tempCnt + 1;
                end
            end
            
            if (rem(numFiles,numSlices)~=0)
              disp('Dicom Number Error');
            end
            
        end
    end
end
