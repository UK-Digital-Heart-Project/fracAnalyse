function [ FormerBinaryMasksPresent, BinaryMaskStack, ReturnCode ] = pft_FindPreviousBinaryMasks(TopLevelOutputFolder, LeafFolder)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A function to retrieve and re-use already created binary masks (in case the code for the FD calculation should change.             %
%                                                                                                                                    %
% The o/p files sought for are in fullfile(TopLevelOutputFolder, 'ThresImg', LeafFolder),                                           %
% and have names such as binaryMaskSlice1Phase1.png.                                                                                 %
%                                                                                                                                    %
% Outputs are:                                                                                                                       %
%                                                                                                                                    %
% FormerBinaryMasksPresent: a logical vector, one entry for each (possible) slice, numbered 1..20, but First and Last will be found. %
% BinaryMaskStack:          a 3-D logical image stack.                                                                               %
% ReturnCode:               a string with values { 'Folder for search does not exist.', ...                                          %
%                                                  'No former ROI''s found.', ...                                                    %
%                                                  'Binary mask stack not contiguous.' ...                                           %
%                                                  'OK' }.                                                                           %
% These are self-explanatory: (1) and (3) shouldn't happen in practice, (2) is possible and (4) means 'Success - go ahead.'.         %
%                                                                                                                                    %
% PFT - 01. 12. 2016.                                                                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A global constant which matches the auto-FD code, but any number large enough in practice will suffice
NSLICES = 20;

% Default o/p for immediate return
FormerBinaryMasksPresent = false([1, NSLICES]);
BinaryMaskStack = [];
ReturnCode = 'Folder for search does not exist.';

% Check that the folder to be searched actually exists
if (exist(fullfile(TopLevelOutputFolder, 'ThresImg', LeafFolder), 'dir') ~= 7)  
  return;
end

% Check for the presence of end-diastole files only
for n = 1:NSLICES
  FileName = sprintf('binaryMaskSlice%1dPhase1.png', n);
  if (exist(fullfile(TopLevelOutputFolder, 'ThresImg', LeafFolder, FileName), 'file') == 2)
    FormerBinaryMasksPresent(n) = true;
  end
end

% Return if no previous ROI's are found
if (sum(FormerBinaryMasksPresent) == 0)
  ReturnCode = 'No former ROI''s found.';
  return;
end  

% Now check for contiguity
First = find(FormerBinaryMasksPresent == true, 1, 'first');
Last  = find(FormerBinaryMasksPresent == true, 1, 'last');

M = Last - First + 1;
N = sum(FormerBinaryMasksPresent(First:Last));

if (M ~= N)
  ReturnCode = 'Binary mask stack not contiguous.';
  return;
end

% Finally, extract the stack of binary masks - use the 'first' one present to determine the image dimensions
FileName = sprintf('binaryMaskSlice%1dPhase1.png', First);
PathName = fullfile(TopLevelOutputFolder, 'ThresImg', LeafFolder, FileName);
BM = logical(imread(PathName));

[ NR, NC ] = size(BM);

BinaryMaskStack = false([NR, NC, NSLICES]);

for p = First:Last
  if (p == First)
    BinaryMaskStack(:, :, p) = BM;
  else
    FileName = sprintf('binaryMaskSlice%1dPhase1.png', p);
    PathName = fullfile(TopLevelOutputFolder, 'ThresImg', LeafFolder, FileName);
    BM = logical(imread(PathName));
    BinaryMaskStack(:, :, p) = BM;
  end
end

ReturnCode = 'OK';
    
end

