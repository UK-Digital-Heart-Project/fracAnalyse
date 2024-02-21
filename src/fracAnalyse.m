function varargout = fracAnalyse(varargin)
% FRACANALYSE Fractal Analysis GUI for Fractal Dimension Computation
% Copyright (c) 2016 Cai Jiashen

% FRACANALYSE MATLAB code for fracAnalyse.fig
%      FRACANALYSE, by itself, creates a new FRACANALYSE or raises the existing
%      singleton*.
%
%      H = FRACANALYSE returns the handle to a new FRACANALYSE or the handle to
%      the existing singleton*.
%
%      FRACANALYSE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FRACANALYSE.M with the given input arguments.
%
%      FRACANALYSE('Property','Value',...) creates a new FRACANALYSE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before fracAnalyse_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to fracAnalyse_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help fracAnalyse

% Last Modified by GUIDE v2.5 12-Apr-2016 22:03:49

clearvars -except varargin;
clc;

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @fracAnalyse_OpeningFcn, ...;k
    'gui_OutputFcn',  @fracAnalyse_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before fracAnalyse is made visible.
function fracAnalyse_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to fracAnalyse (see VARARGIN)

% Choose default command line output for fracAnalyse
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% Deactivate PushButton till Folder Load
set(handles.hPushButtonLoadStudy,'Enable','off');
set(handles.hPushButtonReanalyseAll,'Enable','off');
%set(handles.hPushButtonPreload,'Enable','off');
set(handles.hPushButtonE,'Enable','off');
set(handles.hPushButtonP,'Enable','off');
set(handles.hPushButtonCompFD,'Enable','off');
set(handles.hPushButtonFlipSliceLoc,'Enable','off');
set(handles.hSliderHort,'Enable','off');
set(handles.hSliderVert,'Enable','off');
set(handles.hPushButtonReuseMask,'Enable','off');
set(handles.hPushButtonSaveMask,'Enable','off');
set(handles.hPushButtonLockPhase,'Enable','off');
set(handles.hRadioButtonLV,'Enable','off');
set(handles.hRadioButtonRV,'Enable','off');


% Set Current ROI
currentROI = 0;
setappdata(handles.hFig,'curROI',currentROI);

% Add Listeners
addlistener(handles.hSliderHort,'ContinuousValueChange',@(hObject,eventdata) imgRefresh(hObject, eventdata, handles));
addlistener(handles.hSliderVert,'ContinuousValueChange',@(hObject,eventdata) imgRefresh(hObject, eventdata, handles));


% UIWAIT makes fracAnalyse wait for user response (see UIRESUME)
% uiwait(handles.hFig);


% --- Outputs from this function are returned to the command line.
function varargout = fracAnalyse_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% SLIDERS

% --- Executes on slider movement.
function hSliderHort_Callback(hObject, eventdata, handles)
% hObject    handle to hSliderHort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
imgRefresh(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function hSliderHort_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hSliderHort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on slider movement.
function hSliderVert_Callback(hObject, eventdata, handles)
% hObject    handle to hSliderVert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
imgRefresh(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function hSliderVert_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hSliderVert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% PUSHBUTTONS

% --- Executes on button press in hPushButtonLoadFold.
function hPushButtonLoadFold_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonLoadFold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ispc
    homeDir = [getenv('HOMEDRIVE') getenv('HOMEPATH')];
else
    homeDir = getenv('HOME');
end

filepath = uigetdir(homeDir, 'Select directory containing study DICOMs');

if (filepath == 0)
    return;
end

folderList = dir(filepath);
folderList = folderList(arrayfun(@(x) ~strcmp(x.name(1),'.'),folderList));
folderList = folderList(arrayfun(@(x) ~strcmp(x.name,'ThresImg'),folderList));

folderList = folderList([folderList.isdir]);

set(handles.hListBox,'String',{folderList.name});

preloadDone = 0;

% Activate Push Buttons on Folder Load
set(handles.hPushButtonLoadStudy,'Enable','on');
set(handles.hPushButtonReanalyseAll,'Enable','on');
set(handles.hRadioButtonLV,'Enable','on');
set(handles.hRadioButtonRV,'Enable','on');

% Deactivated Preload - JS - 2 Dec 16
%set(handles.hPushButtonPreload,'Enable','on');

setappdata(handles.hFig,'Preload',preloadDone);
setappdata(handles.hFig,'filepath',filepath);
setappdata(handles.hFig,'folderList',folderList);

if ~exist(fullfile(filepath,'ThresImg'),'file')
    mkdir (fullfile(filepath,'ThresImg'));
end

if ~exist(fullfile(filepath,'FDSummary.csv'),'file')
    fileID = fopen(fullfile(filepath,'FDSummary.csv'),'wt');
    headerOut = {'ResearchID', 'SlicesEvaluated', 'SlicesUsed', 'FDMode'...
        'Global_FD', 'Mean_Apical_FD', 'Max_Apical_FD', 'Mean_Basal_FD', 'Max_Basal_FD', ...
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'};
    fprintf(fileID, '%s,', headerOut{1,1:end-1}) ;
    fprintf(fileID, '%s\n', headerOut{1,end}) ;
    fclose(fileID);
end

inputFileType = questdlg('Input Filetype: *.dcm or *.nii.gz','Input Filetype','DICOM','NIFTI','default');
setappdata(handles.hFig,'InputFileType',inputFileType);

% --- Executes on button press in hPushButtonLoadStudy.
function hPushButtonLoadStudy_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonLoadStudy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get Selected Study
selectedStudyNo = get(handles.hListBox,'Value');
currentStudyNo = selectedStudyNo;

preloadDone = getappdata(handles.hFig,'Preload');

inputFileType = getappdata(handles.hFig,'InputFileType');

if (preloadDone == 0)
    filepath = getappdata(handles.hFig,'filepath');
    folderList = getappdata(handles.hFig,'folderList');
    
    %set(handles.hPushButtonLoadStudy,'Enable','off');
    pause(0.01);
    
    % PFT - edited 24/11/2016 % JS - edited 2/12/2016 **/*.dcm -> **/*
    
    if strcmp(inputFileType,'DICOM')
        tWaitBar=waitbar(0.5,'Loading DICOMs...');
        fileList = rdir(fullfile(filepath,folderList(currentStudyNo).name,'**',filesep,'*.dcm'));%.dcm
        fileList = fileList(~[fileList.isdir]);
        
        if numel(fileList) > 0
            [~, ~, tempext] = fileparts(fileList(1).name);
            if strcmpi(tempext,'.DS_STORE')
                fileList(1)=[];
                disp('removed .DS_STORE');
            end
        end
        
        if numel(fileList) ~= 0
            tic;
            %imgData=pft_FD_ReadDicomImageCineStack(fullfile(filepath,folderList(currentStudyNo).name));
            
            [imgData, voxelSize]=loadDicom(fullfile(filepath,folderList(currentStudyNo).name),0);
            toc
        else
            msgbox('DICOMs not found in folder','Error');
            set(handles.hPushButtonLoadStudy,'Enable','on');
            pause(0.01);
            
            close(tWaitBar);
            return;
        end
        
        set(handles.hPushButtonLoadStudy,'Enable','on');
        pause(0.01);
        
        close(tWaitBar);
        [~,~,maxZ,maxT] = size(imgData);
        
    elseif strcmp(inputFileType,'NIFTI')
        tWaitBar=waitbar(0.5,'Loading NIFTI...');
        
        checkNIFTI = exist(fullfile(filepath,folderList(currentStudyNo).name,'sa_ED.nii.gz'), 'file');
             
        fullfile(filepath,folderList(currentStudyNo).name,'sa_ED.nii.gz')
        if checkNIFTI == 2
            tic;
            inputNIFTIData = load_untouch_nii(fullfile(filepath,folderList(currentStudyNo).name,'sa_ED.nii.gz'));
            imgData = inputNIFTIData.img;
            voxelSize = inputNIFTIData.hdr.dime.pixdim(2:3);
            toc
        else
            msgbox('NIFTI not found in folder','Error');
            set(handles.hPushButtonLoadStudy,'Enable','on');
            pause(0.01);
            
            close(tWaitBar);
            return;
        end
        
        set(handles.hPushButtonLockPhase,'String','Unlock Phase');
        set(handles.hSliderHort,'Enable','off');
        pause(0.01);
        
        close(tWaitBar);
        
        [~,~,maxZ] = size(imgData);
        maxT = 1;
    end
% Preload Disabled 
%{ 
elseif (preloadDone == 1)
    imgDataPreload = getappdata(handles.hFig,'ImgDataPreload');
    folderList = getappdata(handles.hFig,'folderList');
    
    strcurrfolder = ['I' folderList(currentStudyNo).name];
    strcurrfolder = strrep(strcurrfolder,'-','_');
    
    try
        imgData = imgDataPreload{currentStudyNo};
        voxelSize = voxelSize{currentStudyNo};
    catch
        filepath = getappdata(handles.hFig,'filepath');
        folderList = getappdata(handles.hFig,'folderList');
        
        set(handles.hPushButtonLoadStudy,'Enable','off');
        pause(0.01);
        
        tWaitBar=waitbar(0.5,'Loading DICOMs...');
        
        fileList = rdir(fullfile(filepath,folderList(currentStudyNo).name,'**/*'));%.dcm
        fileList = fileList(~[fileList.isdir]);
        
        if numel(fileList) ~= 0
            [imgData, voxelSize]=loadDicom(fullfile(filepath,folderList(currentStudyNo).name),0);
        else
            msgbox('DICOMs not found in folder','Error');
            set(handles.hPushButtonLoadStudy,'Enable','on');
            pause(0.01);
            
            close(tWaitBar);
            return;
        end
        drawnow;
        set(handles.hPushButtonLoadStudy,'Enable','on');
        pause(0.01);
        
        close(tWaitBar);
    end
%}
end

% Set Slider Handles

if maxT == 1
    set(handles.hSliderHort,'Enable','off');
    set(handles.hSliderHort,'Min',1,'Max',1,'Value',1);
else
    set(handles.hSliderHort,'Min',1,'Max',maxT,'Value',1,'SliderStep',[1/(maxT-1) 2/(maxT-1)]);
end

set(handles.hSliderVert,'Min',1,'Max',maxZ,'Value',1,'SliderStep',[1/(maxZ-1) 2/(maxZ-1)]);

% Set Textboxes
set(handles.hTextCurrStudy,'String',sprintf('Subject ID: %s',folderList(currentStudyNo).name));
set(handles.hEditT,'String','1');
set(handles.hEditZ,'String','1');
set(handles.hTextROIIndicator,'String','');


% Activate Pushbutton and Slider
set(handles.hPushButtonE,'Enable','on');
set(handles.hPushButtonP,'Enable','on');
set(handles.hPushButtonCompFD,'Enable','on');
set(handles.hPushButtonFlipSliceLoc,'Enable','on');
set(handles.hPushButtonReuseMask,'Enable','on');
set(handles.hPushButtonReuseMask,'String','Load ROIs','BackgroundColor', [0.94 0.94 0.94]);
set(handles.hPushButtonSaveMask,'Enable','on');
set(handles.hPushButtonLockPhase,'Enable','on');

set(handles.hSliderVert,'Enable','on');

currentState = get(handles.hPushButtonLockPhase,'String');

if strcmp(currentState,'Lock Phase')
    set(handles.hSliderHort,'Enable','on');
else
    set(handles.hSliderHort,'Enable','off');
end

% Load Image
imshow(imadjust(mat2gray(imgData(:,:,1,1))),'Parent',handles.hAxes);

% Pre-allocate Variables
grayIData = {};
cropIData = {};
thresIData = {};
binaryMaskData = {};
binaryMaskFullData = {};
dataZ = [];
dataT = [];
fdData = [];
slicesFlipStatus = false;

% Store in Handles
setappdata(handles.hFig,'ImgData',imgData);
setappdata(handles.hFig,'TotalPhases',maxT);
setappdata(handles.hFig,'TotalSlices',maxZ);
setappdata(handles.hFig,'VoxelSize',voxelSize);
setappdata(handles.hFig,'GrayImgData',grayIData);
setappdata(handles.hFig,'CropImgData',cropIData);
setappdata(handles.hFig,'ThresImgData',thresIData);
setappdata(handles.hFig,'BinaryMaskData',binaryMaskData);
setappdata(handles.hFig,'BinaryMaskFullData',binaryMaskFullData);
setappdata(handles.hFig,'SliceLocData',dataZ);
setappdata(handles.hFig,'PhaseData',dataT);
setappdata(handles.hFig,'FDData',fdData);
setappdata(handles.hFig,'CurrentStudyNo',currentStudyNo);
setappdata(handles.hFig,'SlicesFlipStatus',slicesFlipStatus);



% --- Executes on button press in hPushButtonReanalyseAll.
function hPushButtonReanalyseAll_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonReanalyseAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
filepath = getappdata(handles.hFig,'filepath');
folderList = getappdata(handles.hFig,'folderList');

inputFileType = getappdata(handles.hFig,'InputFileType');

set(handles.hPushButtonReanalyseAll,'Enable','off');
pause(0.01);

tWaitBar = waitbar(0,'Analysing Cases...','Visible','off');

posWaitBar = get(tWaitBar,'Position');
posWaitBar(2)  = posWaitBar(2) + 2 * posWaitBar(4);

set(tWaitBar,'Position',posWaitBar,'Visible','on');

flagError = 0;
folderError = {};

for foldercnt = 1:numel(folderList)
    
    waitbar(foldercnt/numel(folderList),tWaitBar,sprintf('Analysing Case %i/%i',foldercnt,numel(folderList)));
    
    
    if strcmp(inputFileType,'DICOM')
        
        fileList = rdir(fullfile(filepath,folderList(foldercnt).name,'**/*.dcm'));
        fileList = fileList(~[fileList.isdir]);
        
        if numel(fileList) > 0
            [~, ~, tempext] = fileparts(fileList(1).name);
            if strcmpi(tempext,'.DS_STORE')
                fileList(1)=[];
                disp('removed .DS_STORE');
            end
        end
        
        if numel(fileList) ~= 0
            [imgData, voxelSize]= loadDicom(fullfile(filepath,folderList(foldercnt).name),0);
            [~,~,maxZ,maxT] = size(imgData);
        end
        
        
    elseif strcmp(inputFileType,'NIFTI')    
            checkNIFTI = exist(fullfile(filepath,folderList(foldercnt).name,'sa_ED.nii.gz'), 'file');
    
            if checkNIFTI == 2
                inputNIFTIData = load_nii(fullfile(filepath,folderList(foldercnt).name,'sa_ED.nii.gz'));
                imgData = inputNIFTIData.img;
                voxelSize = inputNIFTIData.hdr.dime.pixdim(2:3);
            else
                msgbox('NIFTI not found in folder','Error');
                set(handles.hPushButtonLoadStudy,'Enable','on');
                pause(0.01);
    
                return;
            end
    
            set(handles.hPushButtonLockPhase,'String','Unlock Phase');
            set(handles.hSliderHort,'Enable','off');
            pause(0.01);
        
            [~,~,maxZ] = size(imgData);
            maxT=1;
            set(handles.hPushButtonLockPhase,'String','Unlock Phase');
            set(handles.hSliderHort,'Enable','off');
        
    end
    
    currentStudyNo = foldercnt;
    
    % Set Slider Handles
    if maxT == 1
        set(handles.hSliderHort,'Enable','off');
        set(handles.hSliderHort,'Min',1,'Max',1,'Value',1);
    else
        set(handles.hSliderHort,'Min',1,'Max',maxT,'Value',1,'SliderStep',[1/(maxT-1) 2/(maxT-1)]);
    end
    set(handles.hSliderVert,'Min',1,'Max',maxZ,'Value',1,'SliderStep',[1/(maxZ-1) 2/(maxZ-1)]);
    
    % Set Textboxes
    set(handles.hTextCurrStudy,'String',sprintf('Subject ID: %s',folderList(currentStudyNo).name));
    set(handles.hEditT,'String','1');
    set(handles.hEditZ,'String','1');
    set(handles.hTextROIIndicator,'String','');
    
    
    % Activate Pushbutton and Slider
    set(handles.hPushButtonE,'Enable','on');
    set(handles.hPushButtonP,'Enable','on');
    set(handles.hPushButtonCompFD,'Enable','on');
    set(handles.hPushButtonFlipSliceLoc,'Enable','on');
    set(handles.hPushButtonReuseMask,'Enable','on');
    set(handles.hPushButtonReuseMask,'String','Load ROIs','BackgroundColor', [0.94 0.94 0.94]);
    
    set(handles.hSliderVert,'Enable','on');
    
    currentState = get(handles.hPushButtonLockPhase,'String');
    
    if strcmp(currentState,'Lock Phase')
        set(handles.hSliderHort,'Enable','on');
    else
        set(handles.hSliderHort,'Enable','off');
    end
    
    % Load Image
    imshow(imadjust(mat2gray(imgData(:,:,1,1))),'Parent',handles.hAxes);
    
    % Pre-allocate Variables
    grayIData = {};
    cropIData = {};
    thresIData = {};
    binaryMaskData = {};
    binaryMaskFullData = {};
    dataZ = [];
    dataT = [];
    fdData = [];
    slicesFlipStatus = false;
        
    % Store in Handles
    setappdata(handles.hFig,'ImgData',imgData);
    setappdata(handles.hFig,'TotalPhases',maxT);
    setappdata(handles.hFig,'TotalSlices',maxZ);
    setappdata(handles.hFig,'VoxelSize',voxelSize);
    setappdata(handles.hFig,'GrayImgData',grayIData);
    setappdata(handles.hFig,'CropImgData',cropIData);
    setappdata(handles.hFig,'ThresImgData',thresIData);
    setappdata(handles.hFig,'BinaryMaskData',binaryMaskData);
    setappdata(handles.hFig,'BinaryMaskFullData',binaryMaskFullData);
    setappdata(handles.hFig,'SliceLocData',dataZ);
    setappdata(handles.hFig,'PhaseData',dataT);
    setappdata(handles.hFig,'FDData',fdData);
    setappdata(handles.hFig,'CurrentStudyNo',currentStudyNo);
    setappdata(handles.hFig,'SlicesFlipStatus',slicesFlipStatus);
    
    statusCode = loadMasks(hObject, eventdata, handles);
    
    if statusCode == false
        flagError = flagError + 1;
        folderError{end+1} = folderList(foldercnt).name;
    else
        modifierCalledFunc = true;
        hPushButtonCompFD_Callback(hObject, eventdata, handles, modifierCalledFunc);
    end
end

close(tWaitBar);

waitfor(msgbox(sprintf('Re-Analysis Complete for %i/%i Cases',numel(folderList)-flagError,numel(folderList))));

if flagError > 0
    waitfor(msgbox(['Error in folders:',sprintf('\n%s',folderError{:})],'Error'));   
end


%{
% --- Executes on button press in hPushButtonPreload.
function hPushButtonPreload_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonPreload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

filepath = getappdata(handles.hFig,'filepath');
folderList = getappdata(handles.hFig,'folderList');

set(handles.hPushButtonPreload,'Enable','off');
pause(0.01);

tWaitBar=parfor_progressbar(numel(folderList),'Loading DICOMs...');

flagDicomNotFound = 0;
folderDicomNotFound = {};

numel(folderList)

for foldercnt = 1:numel(folderList)
%     strcurrfolder = ['I' folderList(foldercnt).name];
%     strcurrfolder = strrep(strcurrfolder,'-','_');
tic

    fileList = rdir(fullfile(filepath,folderList(foldercnt).name,'**/*.dcm'));
    fileList = fileList(~[fileList.isdir]);
    
    if numel(fileList) > 0
        [~, ~, tempext] = fileparts(fileList(1).name);
        if strcmpi(tempext,'.DS_STORE')
            fileList(1)=[];
            disp('removed .DS_STORE');
        end
    end
    
    if numel(fileList) ~= 0
        [imgDataPreload{foldercnt} voxelSizePreload{foldercnt}]= loadDicom(fullfile(filepath,folderList(foldercnt).name),0);
    end
    %     else
    %         flagDicomNotFound = flagDicomNotFound + 1;
    %         folderDicomNotFound{end+1} = folderList(foldercnt).name;
    %
    %     end
    %     waitbar(foldercnt/numel(folderList),tWaitBar,sprintf('Loading DICOMs (%d/%d)',foldercnt,numel(folderList)));
    tWaitBar.iterate(1);
end

toc

close(tWaitBar);

if flagDicomNotFound > 0
    waitfor(msgbox(['DICOMs not found in folders:',sprintf('\n%s',folderDicomNotFound{:})],'Error'));
end
    
preloadDone = 1;

set(handles.hPushButtonPreload,'Enable','off');
setappdata(handles.hFig,'ImgDataPreload',imgDataPreload);
setappdata(handles.hFig,'VoxelSizePreload',voxelSizePreload);
setappdata(handles.hFig,'Preload',preloadDone);
%}

% --- Executes on button press in hPushButtonFlipSliceLoc.
function hPushButtonFlipSliceLoc_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonFlipSliceLoc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

imgData = getappdata(handles.hFig,'ImgData');

imgData = flip(imgData,3);

slicesFlipStatus = getappdata(handles.hFig,'SlicesFlipStatus');
slicesFlipStatus = ~slicesFlipStatus;
setappdata(handles.hFig,'SlicesFlipStatus',slicesFlipStatus);

setappdata(handles.hFig,'ImgData',imgData);
imgRefresh(hObject, eventdata, handles);

% --- Executes on button press in hPushButtonE ROI (Ellipse).
function hPushButtonE_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

currentROI = 1;
setappdata(handles.hFig,'curROI',currentROI);

imgData = getappdata(handles.hFig,'ImgData');
voxelSize = getappdata(handles.hFig,'VoxelSize');
grayIData = getappdata(handles.hFig,'GrayImgData');
cropIData = getappdata(handles.hFig,'CropImgData');
binaryMaskData = getappdata(handles.hFig,'BinaryMaskData');
binaryMaskFullData = getappdata(handles.hFig,'BinaryMaskFullData');
dataZ = getappdata(handles.hFig,'SliceLocData');
dataT = getappdata(handles.hFig,'PhaseData');

currentT = round((get(handles.hSliderHort,'Value')));
currentZ = round((get(handles.hSliderVert,'Value')));

% Check if Previous ROI Set
if (ismember(currentZ,dataZ) && dataT(find(dataZ==currentZ))==currentT)
    [cropIData{find(dataZ == currentZ)}, binaryMaskData{find(dataZ==currentZ)},binaryMaskFullData{find(dataZ==currentZ)}] = roiSelect(imgData(:,:,currentZ,currentT),voxelSize,'ellipse');
    grayIData{find(dataZ == currentZ)} = imgData(:,:,currentZ,currentT);
    
else
    [cropIData{end+1}, binaryMaskData{end+1},binaryMaskFullData{end+1}] = roiSelect(imgData(:,:,currentZ,currentT),voxelSize,'ellipse');
    grayIData{end+1} = imgData(:,:,currentZ,currentT);
    
    dataT(end+1) = currentT;
    dataZ(end+1) = currentZ;
end

if (ismember(currentZ,dataZ) && dataT(find(dataZ==currentZ))==currentT)
    set(handles.hTextROIIndicator,'String','ROI Set');
else
    set(handles.hTextROIIndicator,'String','');
end

currentROI = 0;
setappdata(handles.hFig,'curROI',currentROI);

% Store in Handles
setappdata(handles.hFig,'GrayImgData',grayIData);
setappdata(handles.hFig,'CropImgData',cropIData);
setappdata(handles.hFig,'BinaryMaskData',binaryMaskData);
setappdata(handles.hFig,'BinaryMaskFullData',binaryMaskFullData);
setappdata(handles.hFig,'SliceLocData',dataZ);
setappdata(handles.hFig,'PhaseData',dataT);

imgRefresh(hObject, eventdata, handles);


% --- Executes on button press in hPushButtonP ROI (Polygon).
function hPushButtonP_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonP (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Set currentROI
currentROI = 1;
setappdata(handles.hFig,'curROI',currentROI);

imgData = getappdata(handles.hFig,'ImgData');
voxelSize = getappdata(handles.hFig,'VoxelSize');
grayIData = getappdata(handles.hFig,'GrayImgData');
cropIData = getappdata(handles.hFig,'CropImgData');
binaryMaskData = getappdata(handles.hFig,'BinaryMaskData');
binaryMaskFullData = getappdata(handles.hFig,'BinaryMaskFullData');
dataZ = getappdata(handles.hFig,'SliceLocData');
dataT = getappdata(handles.hFig,'PhaseData');

currentT = round((get(handles.hSliderHort,'Value')));
currentZ = round((get(handles.hSliderVert,'Value')));

% Check if Previous ROI Set
if (ismember(currentZ,dataZ) && dataT(find(dataZ==currentZ))==currentT)
    [cropIData{find(dataZ == currentZ)}, binaryMaskData{find(dataZ==currentZ)}, binaryMaskFullData{find(dataZ==currentZ)}] = roiSelect(imgData(:,:,currentZ,currentT),voxelSize,'poly');
    grayIData{find(dataZ == currentZ)} = imgData(:,:,currentZ,currentT);
    
else
    [cropIData{end+1}, binaryMaskData{end+1}, binaryMaskFullData{end+1}] = roiSelect(imgData(:,:,currentZ,currentT),voxelSize,'poly');
    grayIData{end+1} = imgData(:,:,currentZ,currentT);
    
    dataT(end+1) = currentT;
    dataZ(end+1) = currentZ;
end

if (ismember(currentZ,dataZ) && dataT(find(dataZ==currentZ))==currentT)
    set(handles.hTextROIIndicator,'String','ROI Set');
else
    set(handles.hTextROIIndicator,'String','');
end

% Set currentROI
currentROI = 0;
setappdata(handles.hFig,'curROI',currentROI);

% Store in Handles
setappdata(handles.hFig,'GrayImgData',grayIData);
setappdata(handles.hFig,'CropImgData',cropIData);
setappdata(handles.hFig,'BinaryMaskData',binaryMaskData);
setappdata(handles.hFig,'BinaryMaskFullData',binaryMaskFullData);
setappdata(handles.hFig,'SliceLocData',dataZ);
setappdata(handles.hFig,'PhaseData',dataT);

imgRefresh(hObject, eventdata, handles);


% --- Executes on button press in hPushButtonCompFD.
function hPushButtonCompFD_Callback(hObject, eventdata, handles, modifierCalledFunc)
% hObject    handle to hPushButtonCompFD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

sigma = 4;
epsilon = 3;

tWaitBar = waitbar(0,'Loading Data...');

% Load from Handles
filepath = getappdata(handles.hFig,'filepath');
folderList = getappdata(handles.hFig,'folderList');
currentStudyNo = getappdata(handles.hFig,'CurrentStudyNo');

cropIData = getappdata(handles.hFig,'CropImgData');
binaryMaskData = getappdata(handles.hFig,'BinaryMaskData');
binaryMaskFullData = getappdata(handles.hFig,'BinaryMaskFullData');
dataZ = getappdata(handles.hFig,'SliceLocData');
dataT = getappdata(handles.hFig,'PhaseData');
slicesFlipStatus = getappdata(handles.hFig,'SlicesFlipStatus');



fdMode = 'LV';
% get(handles.hRadioButtonLV,'value')
% get(handles.hRadioButtonRV,'value')

if (get(handles.hRadioButtonLV,'value')==1)
    fdMode = 'LV';
elseif (get(handles.hRadioButtonRV,'value')==1)
    fdMode = 'RV';
end

set(handles.hPushButtonCompFD,'Enable','off')
pause(0.01);

waitbar(0.25,tWaitBar,'Fractal Analysis...');

% Sort Index
[~, sortIndex] = sort(dataZ);


% Check if ROI set
if isempty(dataZ)
    h = msgbox('ROI not set','Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

% Check for Missing Slices
if all(diff(dataZ(sortIndex)) < 3) == 0
    regSlice=sprintf('%d ', dataZ(sortIndex));
    h = msgbox(['Slices not in order',sprintf('\nRegistered Slices: %s',regSlice)] ,'Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

% Check for Missing Slices
if all(diff(dataZ(sortIndex)) < 3) == 0
    regSlice=sprintf('%d ', dataZ(sortIndex));
    h = msgbox(['Slices not in order',sprintf('\nRegistered Slices: %s',regSlice)] ,'Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

% Preallocate variables
thresIData=cell(numel(cropIData));
fdData= zeros(numel(cropIData));
bcFigData = gobjects(numel(cropIData));

parfor sliceCnt = 1:numel(cropIData) % TDedit: changed parfor to for
     try
        [fdData(sliceCnt), thresIData{sliceCnt}, bcFigData(sliceCnt)] = fracDimBatch(imadjust(cropIData{sliceCnt}),binaryMaskData{sliceCnt},sigma,epsilon,fdMode);
     catch
        fdData(sliceCnt) = 0.0 % Zero for error
        thresIData{sliceCnt} = [];
        bcFigData(sliceCnt) = gobjects(0);
        msgbox('Error');
     end
end

waitbar(0.5,tWaitBar,'Exporting Results...');

minROISlice = min(dataZ);

fdOutput ={folderList(currentStudyNo).name};

% Compute FD statistics

discardMode = false;
fdStats=fdStatistics(fdData(sortIndex),discardMode);

% Changed output precision from .3/.5f to .9f JC 14/1/17

fdOutput = [fdOutput {sprintf('%d, %d, %s',fdStats.evalSlices, fdStats.usedSlices, fdMode)}];
fdOutput = [fdOutput {sprintf('%0.9f, %0.9f, %0.9f, %0.9f, %0.9f',fdStats.globalFD, fdStats.meanApicalFD, fdStats.maxApicalFD, fdStats.meanBasalFD, fdStats.maxBasalFD)}];


% PFT - noted on 24/11/2016
if minROISlice > 1
    for n = 1:minROISlice-1
        fdOutput = [fdOutput {''}];
    end
end

% Sort slices in ascending order (SliceLoc)
for sliceCnt = sortIndex
    fprintf('Slice = %d, Time = %d, FD = %0.3f\n',dataZ(sliceCnt), dataT(sliceCnt), fdData(sliceCnt));
    fdOutput = [fdOutput {sprintf('%0.5f',fdData(sliceCnt))}];
end


fileID = fopen(fullfile(filepath,'FDSummary.csv'),'at');

fprintf(fileID, '%s,', fdOutput{1,1:end-1}) ;
fprintf(fileID, '%s\n', fdOutput{1,end}) ;
fclose(fileID);

clear fdOutput;

waitbar(0.75,tWaitBar,'Exporting Images...');

if ~exist(fullfile(filepath,'ThresImg',sprintf('%s',folderList(currentStudyNo).name)),'file')
    mkdir (fullfile(filepath,'ThresImg',sprintf('%s',folderList(currentStudyNo).name)));
end

subjName = sprintf('%s',folderList(currentStudyNo).name);

fileID = fopen(fullfile(filepath,'ThresImg',subjName,'ThresDetails.csv'),'wt');
thresSummaryOutput = sprintf('SlicesFlipStatus\n%d', slicesFlipStatus);
fprintf(fileID, thresSummaryOutput);
fclose(fileID);

THRESHOLD = 0.1;

for sliceCnt = sortIndex
    if (fdData(sliceCnt) < THRESHOLD) % PFT - skip image import if FD for current slice failed
        continue;
    end
    
    fprintf('%0.5f\t',fdData(sliceCnt));
    
    % Export Thresholded Images with Contour (Optional)
    thresI = thresIData{sliceCnt};
    cropI = cropIData{sliceCnt};
    binaryMaskFull = binaryMaskFullData{sliceCnt};

    [~, threshold] = edge(thresI,'sobel');
    edgeI = edge(thresI,'sobel',threshold*0.5);
    
    cropI = interp2(cropI,3,'cubic');
    edgeI = interp2(double(edgeI),3,'cubic');
    
    filename = sprintf('thresSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));
    imwrite(imadd(cropI,edgeI),fullfile(filepath,'ThresImg',subjName,filename));
    
    % Write binary mask
    filename = sprintf('binaryMaskSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));

    imwrite(logical(binaryMaskFull),fullfile(filepath,'ThresImg',subjName,filename));
    
    
    %Export cropped images (Optional)
    filename = sprintf('cropSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));

    imwrite(cropI,fullfile(filepath,'ThresImg',subjName,filename));
    
    filename = sprintf('edgeSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));

    imwrite(edgeI,fullfile(filepath,'ThresImg',subjName,filename));
    
    filename = sprintf('bcPlotSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));
    
    export_fig(bcFigData(sliceCnt), fullfile(filepath,'ThresImg',subjName,filename), '-png', '-m3');         
    
    delete(bcFigData(sliceCnt));
    
    clear cropI;    
    clear thresI;
    clear edgeI;
    clear filename;
end

fprintf('\n');

close(tWaitBar);

set(handles.hPushButtonCompFD,'Enable','on')
pause(0.01);

if nargin < 4
    msgbox('Compute FD Done','Done');
end
% Store in Handles
setappdata(handles.hFig,'ThresImgData',thresIData);
setappdata(handles.hFig,'FDData',fdData);

function hPushButtonReuseMask_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonReuseMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
loadMasks(hObject, eventdata, handles)

function hPushButtonSaveMask_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonReuseMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

tWaitBar = waitbar(0,'Loading Data...');

% Load from Handles
filepath = getappdata(handles.hFig,'filepath');
folderList = getappdata(handles.hFig,'folderList');
currentStudyNo = getappdata(handles.hFig,'CurrentStudyNo');

cropIData = getappdata(handles.hFig,'CropImgData');
binaryMaskData = getappdata(handles.hFig,'BinaryMaskData');
binaryMaskFullData = getappdata(handles.hFig,'BinaryMaskFullData');
dataZ = getappdata(handles.hFig,'SliceLocData');
dataT = getappdata(handles.hFig,'PhaseData');
slicesFlipStatus = getappdata(handles.hFig,'SlicesFlipStatus');


set(handles.hPushButtonSaveMask,'Enable','off')
pause(0.01);

waitbar(0.5,tWaitBar,'Saving ROIs...');

% Sort Index
[~, sortIndex] = sort(dataZ);

% Check if ROI set
if isempty(dataZ)
    h = msgbox('ROI not set','Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

% Check for Missing Slices
if all(diff(dataZ(sortIndex)) <1) == 0
    regSlice=sprintf('%d ', dataZ(sortIndex));
    h = msgbox(['Slices not in order',sprintf('\nRegistered Slices: %s',regSlice)] ,'Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

% Check for Missing Slices
if all(diff(dataZ(sortIndex)) <1) == 0
    regSlice=sprintf('%d ', dataZ(sortIndex));
    h = msgbox(['Slices not in order',sprintf('\nRegistered Slices: %s',regSlice)] ,'Error');
    close(tWaitBar);
    set(handles.hPushButtonCompFD,'Enable','on')
    return;
end

if ~exist(fullfile(filepath,'ThresImg',sprintf('%s',folderList(currentStudyNo).name)),'file')
    mkdir (fullfile(filepath,'ThresImg',sprintf('%s',folderList(currentStudyNo).name)));
end

subjName = sprintf('%s',folderList(currentStudyNo).name);

fileID = fopen(fullfile(filepath,'ThresImg',subjName,'ThresDetails.csv'),'wt');
thresSummaryOutput = sprintf('SlicesFlipStatus\n%d', slicesFlipStatus);
fprintf(fileID, thresSummaryOutput);
fclose(fileID);

for sliceCnt = sortIndex    
    % Export Thresholded Images with Contour (Optional)
    binaryMaskFull = binaryMaskFullData{sliceCnt};

    % Write binary mask
    filename = sprintf('binaryMaskSlice%dPhase%d.png',dataZ(sliceCnt),dataT(sliceCnt));

    imwrite(logical(binaryMaskFull),fullfile(filepath,'ThresImg',subjName,filename));
   
    clear filename;
end

close(tWaitBar);

set(handles.hPushButtonSaveMask,'Enable','on')
pause(0.01);


function hPushButtonLockPhase_Callback(hObject, eventdata, handles)
% hObject    handle to hPushButtonLockPhase (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

currentState = get(handles.hSliderHort,'Enable');

if strcmp(currentState,'on')
    set(handles.hPushButtonLockPhase,'String','Unlock Phase');
    set(handles.hSliderHort,'Enable','off');
else
    set(handles.hPushButtonLockPhase,'String','Lock Phase');
    set(handles.hSliderHort,'Enable','on');
end

% TEXTBOXES

% --- Executes during object creation, after setting all properties.
function hEditZ_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hEditZ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function hEditT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hEditT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% LISTBOX

% --- Executes on selection change in hListBox.
function hListBox_Callback(hObject, eventdata, handles)
% hObject    handle to hListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns hListBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from hListBox


% --- Executes during object creation, after setting all properties.
function hListBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function hKeyPressFcn(hObject, eventdata, handles)
currentROI = getappdata(handles.hFig,'curROI');
switch eventdata.Key
    case 'uparrow'
        if get(handles.hSliderVert,'Value') < get(handles.hSliderVert,'Max')
            set(handles.hSliderVert,'Value', get(handles.hSliderVert,'Value') + 1);
            imgRefresh(hObject, eventdata, handles);
        end
    case 'w'
        if get(handles.hSliderVert,'Value') < get(handles.hSliderVert,'Max')
            set(handles.hSliderVert,'Value', get(handles.hSliderVert,'Value') + 1);
            imgRefresh(hObject, eventdata, handles);
        end
    case 'downarrow'
        if get(handles.hSliderVert,'Value') > get(handles.hSliderVert,'Min')
            set(handles.hSliderVert,'Value', get(handles.hSliderVert,'Value') - 1);
            imgRefresh(hObject, eventdata, handles);
        end
    case 's'
        if get(handles.hSliderVert,'Value') > get(handles.hSliderVert,'Min')
            set(handles.hSliderVert,'Value', get(handles.hSliderVert,'Value') - 1);
            imgRefresh(hObject, eventdata, handles);
        end
    case 'leftarrow'
        if strcmp(get(handles.hSliderHort,'Enable'),'on')
            if get(handles.hSliderHort,'Value') > get(handles.hSliderHort,'Min')
                set(handles.hSliderHort,'Value', get(handles.hSliderHort,'Value') - 1);
                imgRefresh(hObject, eventdata, handles);
            end
        end
   case 'a'
        if strcmp(get(handles.hSliderHort,'Enable'),'on')
            if get(handles.hSliderHort,'Value') > get(handles.hSliderHort,'Min')
                set(handles.hSliderHort,'Value', get(handles.hSliderHort,'Value') - 1);
                imgRefresh(hObject, eventdata, handles);
            end
        end
    case 'rightarrow'
        if strcmp(get(handles.hSliderHort,'Enable'),'on')
            if get(handles.hSliderHort,'Value') < get(handles.hSliderHort,'Max')
                set(handles.hSliderHort,'Value', get(handles.hSliderHort,'Value') + 1);
                imgRefresh(hObject, eventdata, handles);
            end
        end
    case 'd'
        if strcmp(get(handles.hSliderHort,'Enable'),'on')
            if get(handles.hSliderHort,'Value') < get(handles.hSliderHort,'Max')
                set(handles.hSliderHort,'Value', get(handles.hSliderHort,'Value') + 1);
                imgRefresh(hObject, eventdata, handles);
            end
        end
    case 'e'
        if currentROI == 0
            hPushButtonE_Callback(hObject, eventdata, handles);
        end
    case 'r'
        if currentROI == 0
            hPushButtonP_Callback(hObject, eventdata, handles);
        end
end

% SUPPORTING FUNCTIONS

function imgRefresh(hObject, eventdata, handles)

imgData = getappdata(handles.hFig,'ImgData');
dataZ = getappdata(handles.hFig,'SliceLocData');
dataT = getappdata(handles.hFig,'PhaseData');

currentT = round((get(handles.hSliderHort,'Value')));
currentZ = round((get(handles.hSliderVert,'Value')));

imshow(imadjust(mat2gray(imgData(:,:,currentZ,currentT))),'Parent',handles.hAxes);

set(handles.hEditT,'String',num2str(currentT));
set(handles.hEditZ,'String',num2str(currentZ));

if (ismember(currentZ,dataZ) && dataT(find(dataZ==currentZ))==currentT)
    set(handles.hTextROIIndicator,'String','ROI Set');
    binaryMaskFullData = getappdata(handles.hFig,'BinaryMaskFullData');
    imgData = getappdata(handles.hFig,'ImgData');
    voxelSize = getappdata(handles.hFig,'VoxelSize');
    
    grayImg = imadjust(interpImage(imgData(:,:,currentZ,currentT),voxelSize));
    rgbImg = cat(3,grayImg,grayImg,grayImg);
    
    binaryMask = double(binaryMaskFullData{find(dataZ == currentZ)});
    rgbBinaryMask = cat(3,0.9*binaryMask,0.1*binaryMask,0.3*binaryMask);
    

    overlayImg = imlincomb(0.25,rgbBinaryMask,1,rgbImg);
    
    % overlayI = imadd(rgbI,alphaBinaryMask);
    imshow(overlayImg, 'Parent', handles.hAxes); % Parent added - PFT - 05/12/2016

    
else
    set(handles.hTextROIIndicator,'String','');
end

function targetImg = interpImage (sourceImg,voxelSize)
% Image Grayscaling
targetImg=mat2gray(sourceImg);

[dimRow, dimCol] = size(targetImg);

% disp([dimRow, dimCol]);
% disp(voxelSize);

magFactor = 4; % 1 mm to 4 pixels

interpDimRow = uint16(round(dimRow * voxelSize(1) * magFactor)) ;
interpDimCol = uint16(round(dimCol * voxelSize(2) * magFactor));

disp([interpDimRow, interpDimCol]);

% Image Magnification by Bicubic Interpolation
targetImg=imresize(targetImg,[interpDimRow, interpDimCol], 'bicubic');


function [cropImg, binaryMaskCrop, binaryMaskFull] = roiSelect(sourceImg,voxelSize,mode,reuseMask)

% Image Interpolation
grayImg=interpImage (sourceImg,voxelSize);

% Image Crop
switch mode
    case 'ellipse'
        imshow(imadjust(grayImg));
        
        hROI=imellipse;
        wait(hROI);
        binaryMaskFull = hROI.createMask();
        delete(hROI);
        
    case 'poly'
        imshow(imadjust(grayImg));
        
        hROI=impoly;
        wait(hROI);
        binaryMaskFull = hROI.createMask();
        size(binaryMaskFull)
        delete(hROI);

    case 'reuse'
        binaryMaskFull = reuseMask;
end
        regPropsOutput=regionprops(binaryMaskFull, 'BoundingBox');
        
        cropImg = imcrop(grayImg.*binaryMaskFull, [regPropsOutput.BoundingBox(1) regPropsOutput.BoundingBox(2) regPropsOutput.BoundingBox(3) regPropsOutput.BoundingBox(4)]);
        binaryMaskCrop = imcrop(binaryMaskFull,[regPropsOutput.BoundingBox(1) regPropsOutput.BoundingBox(2) regPropsOutput.BoundingBox(3) regPropsOutput.BoundingBox(4)]);

% Old ROISelect
%{ 
switch mode
    case 'ellipse'
        imshow(imadjust(grayI));
        
        hROI=imellipse;
        wait(hROI);
        position = getPosition(hROI);
        binaryMaskFull = hROI.createMask();
        delete(hROI);
        
        cropI = imcrop(grayI.*binaryMaskFull, position);
        binaryMask = imcrop(binaryMaskFull,position);
        
    case 'poly'
        imshow(imadjust(grayI));
        
        hROI=impoly;
        wait(hROI);
        position = getPosition(hROI);
        binaryMaskFull = hROI.createMask();
        delete(hROI);
        
        minCropX = min(position(:,1));
        maxCropX = max(position(:,1));
        minCropY = min(position(:,2));
        maxCropY = max(position(:,2));
        
        cropI = imcrop(grayI.*binaryMaskFull, [minCropX minCropY maxCropX-minCropX maxCropY-minCropY]);
        binaryMask = imcrop(binaryMaskFull,[minCropX minCropY maxCropX-minCropX maxCropY-minCropY]);
    case 'reuse'
        size(reuseMask)
        binaryMaskFull = reuseMask;
        regPropsOutput=regionprops(binaryMaskFull, 'BoundingBox');
        
        cropI = imcrop(grayI.*binaryMaskFull, [regPropsOutput.BoundingBox(1) regPropsOutput.BoundingBox(2) regPropsOutput.BoundingBox(3) regPropsOutput.BoundingBox(4)]);
        binaryMask = imcrop(binaryMaskFull,[regPropsOutput.BoundingBox(1) regPropsOutput.BoundingBox(2) regPropsOutput.BoundingBox(3) regPropsOutput.BoundingBox(4)]);
        
end
%}

function statusCode = loadMasks(hObject, eventdata, handles)

statusCode = false;

currentStudyNo = getappdata(handles.hFig,'CurrentStudyNo');

filepath = getappdata(handles.hFig,'filepath');
folderList = getappdata(handles.hFig,'folderList');


[formerMaskPresent, reuseMaskStackData, returnCode ] = pft_FindPreviousBinaryMasks(filepath, folderList(currentStudyNo).name);
%msgbox(returnCode);

reuseMaskButtonColour = [0.94 0.94 0.94];

if strcmp(returnCode,'OK')    
    statusCode = true;
    reuseMaskButtonText = 'ROIs Loaded';
    
    thresDetails=tdfread(fullfile(filepath,'ThresImg',folderList(currentStudyNo).name,'ThresDetails.csv'));
    
    slicesFlipStatus = getappdata(handles.hFig,'SlicesFlipStatus');
    
    if slicesFlipStatus ~= logical(thresDetails.SlicesFlipStatus)
        imgData = getappdata(handles.hFig,'ImgData');
        imgData = flip(imgData,3);

        slicesFlipStatus = getappdata(handles.hFig,'SlicesFlipStatus');
        slicesFlipStatus = ~slicesFlipStatus;
        setappdata(handles.hFig,'SlicesFlipStatus',slicesFlipStatus);

        setappdata(handles.hFig,'ImgData',imgData);
        imgRefresh(hObject, eventdata, handles);
    end
    
    % Clear Existing Masks
    grayIData = {};
    cropIData = {};
    binaryMaskData = {};
    binaryMaskFullData = {};
    dataZ = [];
    dataT = [];
    
    
    % Load Img Data
    imgData = getappdata(handles.hFig,'ImgData');
    maxZ = getappdata(handles.hFig,'TotalSlices');
    voxelSize = getappdata(handles.hFig,'VoxelSize');

    % Loop through Slices
    for currentZ = 1:maxZ
        % only ED slices
        currentT = 1;

        if formerMaskPresent(currentZ)
            disp(currentZ);
            [cropIData{end+1}, binaryMaskData{end+1}, binaryMaskFullData{end+1}] = roiSelect(imgData(:,:,currentZ,currentT),voxelSize,'reuse',reuseMaskStackData(:,:,currentZ));
            grayIData{end+1} = imgData(:,:,currentZ,currentT);

            dataT(end+1) = currentT;
            dataZ(end+1) = currentZ;
       end
    end

    setappdata(handles.hFig,'GrayImgData',grayIData);
    setappdata(handles.hFig,'CropImgData',cropIData);
    setappdata(handles.hFig,'BinaryMaskData',binaryMaskData);
    setappdata(handles.hFig,'BinaryMaskFullData',binaryMaskFullData);
    setappdata(handles.hFig,'SliceLocData',dataZ);
    setappdata(handles.hFig,'PhaseData',dataT);

    imgRefresh(hObject, eventdata, handles);

    
elseif strcmp(returnCode,'Folder for search does not exist.')
    reuseMaskButtonText = 'Folder Missing';
    reuseMaskButtonColour = [1 0.5 0.5];
elseif strcmp(returnCode,'No former ROI''s found.')
    reuseMaskButtonText = 'No ROI Found';
    reuseMaskButtonColour = [1 0.5 0.5];
elseif strcmp(returnCode,'Binary mask stack not contiguous.')
    reuseMaskButtonText = 'Non-contiguous';
    reuseMaskButtonColour = [1 0.5 0.5];
end

% Change Button Text & Status
set(handles.hPushButtonReuseMask,'String',reuseMaskButtonText,'BackgroundColor',reuseMaskButtonColour);
set(handles.hPushButtonReuseMask,'Enable','off');


