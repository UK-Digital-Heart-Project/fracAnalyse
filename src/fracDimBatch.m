% fracDimBatch Calculates the fractal dimension of image (thresholding + edge detection)
% Copyright (c) 2016 Cai Jiashen

function [fd, thresI, bcFig] = fracDimBatch(cropI,binaryMask,sigma,epsilon,fdMode)

mag = 100;

%% Level Set Thresholding
% Sets level-set thresholding parameters if not set
if nargin <3
    sigma=4;
    epsilon=3;
end 

% Level set threshold
prethresI = regionfill(cropI,~binaryMask);
%prethresI = roifill(cropI,~binaryMask);

%figure();
%imshow (prethresI);

%prethresI = cropI;

thresI = ~levelSetBatch(prethresI,sigma,epsilon);

thresI=thresI.*binaryMask;

%% Check FD Mode
if strcmp(fdMode,'LV')
    largestAreaMode = 'JS'; % or 'PFT'
elseif strcmp(fdMode,'RV')
    largestAreaMode = 'RV';
end

%% Ellipse around largest area of thresholded image

if strcmp(largestAreaMode,'JS')
    % JS Original ConvexHull/Ellipse
    largestAreaI=bwconvhull(bwpropfilt(imbinarize(thresI,0.5),'Area',1,'largest'));
    statsLA = regionprops(largestAreaI,'Centroid', 'MajorAxisLength','MinorAxisLength','Orientation');
    
    hortLA = statsLA.MajorAxisLength/2;
    vertLA = statsLA.MinorAxisLength/2;
    
    xLA = statsLA.Centroid(1);
    yLA = statsLA.Centroid(2);
    
    [ySize, xSize] = size(largestAreaI);
    
    [xMask, yMask] = meshgrid(1:xSize,1:ySize);
    lAMask = ((xMask-xLA)/hortLA).^2 + ((yMask-yLA)/hortLA).^2 <= 1;
    
    compositeLAI=imadd(largestAreaI,lAMask);
    compositeLAI(compositeLAI>1)=1;

elseif strcmp(largestAreaMode,'PFT')
    % PFT Adaptation of ConvexJull/Ellipse + JS's edits/convention
    largestAreaI = bwconvhull(bwpropfilt(imbinarize(thresI, 0.5), 'Area', 1, 'largest'));
    
    statsLA = regionprops(largestAreaI, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');
    
    xLA = statsLA.Centroid(1);
    yLA = statsLA.Centroid(2);
    hortLA = 1.05*statsLA.MajorAxisLength/2.0;    % Expand the axis by 5 P.C.
    vertLA = 1.05*statsLA.MinorAxisLength/2.0;    % Expand the axis by 5 P.C.
    Theta = (pi/180.0)*statsLA.Orientation;
    cc = cos(Theta);
    ss = sin(Theta);
    
    [ySize, xSize] = size(largestAreaI);
    
    [ xx, yy ] = meshgrid(1:xSize, 1:ySize);
    
    XX = cc*(xx - xLA) - ss*(yy - yLA);
    YY = ss*(xx - xLA) + cc*(yy - yLA);
    
    equivEllipse = (XX.^2/hortLA^2 + YY.^2/vertLA^2 <= 1.0);
    
    delta = round(0.025*(hortLA + vertLA));
    largestAreaI = imdilate(largestAreaI, strel('disk', delta, 4));   % Expand the largest area by roughly the same amount as the ellipse
    
    compositeLAI = largestAreaI | equivEllipse;
elseif strcmp(largestAreaMode,'RV')
    
    % Boundary check for RV segmentation
    countPixels = nnz(thresI);
    invertThresI = (~thresI).*binaryMask;
    countInvertPixels = nnz(invertThresI);

    intensityPixels = sum(sum(cropI.*thresI))/countPixels;
    intensityInvertPixels = sum(sum(cropI.*invertThresI))/countInvertPixels;

    if (intensityInvertPixels>intensityPixels)
        thresI = invertThresI;
        disp('Flip Pixels');
    end
    
    % Largest area
    largestAreaI=bwconvhull(bwpropfilt(imbinarize(thresI,0.5),'Area',1,'largest'));
    delta = 2;
    largestAreaI = imdilate(largestAreaI, strel('disk', delta, 4));
    compositeLAI=largestAreaI;
end

thresI=thresI.*compositeLAI;

%% Edge detection
[~, threshold] = edge(thresI,'sobel');
edgeI = edge(thresI,'sobel',threshold*0.5);


%% Box-counting with FD computation
[fd, bcFig] = bxct(edgeI);

% Overview plots
% figure('Name','Overview','NumberTitle','off');
% subplot(2,2,1), imshow(cropI), title('Gray');
% subplot(2,2,2), imshow(cropI), title('Crop');%, hold on, contour (thresI,'w');
% subplot(2,2,3), imshow(cropI), title(sprintf('Threshold')), hold on, contour (thresI,'w');
% subplot(2,2,4), imshow(edgeI), title(sprintf('Edges, FD = %0.5f',fd));