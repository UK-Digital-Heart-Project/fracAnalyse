function fdStats = fdStatistics(fdData, discardMode)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (c) 2017 Cai Jiashen - proposed changes by PFT - 02/02/2017.                                                              %
% The input array has already been trimmed and contains a contiguous pack of just those slices that have ROI's assigned to them.      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create some default outputs in case of an early return
fdStats = struct('evalSlices', 0, 'usedSlices', 0, ...
                 'globalFD', 0.0, ...
                 'meanBasalFD', 0.0, 'maxBasalFD', 0.0, 'meanApicalFD', 0.0, 'maxApicalFD', 0.0);
         
% Quit if the i/p array is actually empty, perhaps because a particular folder was not processed but there is an entry for it in an XL sheet
if isempty(fdData)
  return;
end
             
% Note the first and last processed slices                
Lower = 1;
Upper = numel(fdData);

% Calculate the first statistic - the number of "evaluated" slices (including NaN's "inside" and zeros anywhere, even at the end)
fdStats.evalSlices = Upper - Lower + 1;

% Return if there are too few values to yield both basal and apical statistics
N = uint32(Upper - Lower + 1);

if (discardMode == false)
  if (N < 2)
    return;
  end
elseif (discardMode == true)
  if (N < 4)
    return;
  end
end

% Trim the data further if end slices are being discarded
if (discardMode == true)
  Lower = Lower + 1;
  Upper = Upper - 1;
  N = N - 2;
end
   
% Trim the working array and assign safely to a new variable
fd = fdData(Lower:Upper);

Q = idivide(N, uint32(2), 'floor');
R = mod(N, uint32(2));

if (R == 0)
  A = 1;
  B = Q;
  C = Q + 1;
  D = N;
elseif (R == 1)
  A = 1;
  B = Q;
  C = Q + 2;
  D = N;
end

% Now create some arrays and calculate the summary statistics, discarding 0.0's and NaN's
GlobalFD = fd;
BasalFD  = fd(A:B);
ApicalFD = fd(C:D);

THRESHOLD = 0.1;    % Avoid comparing a floating-point value to 0.0

GlobalFD(isnan(GlobalFD) | (GlobalFD < THRESHOLD)) = [];    
BasalFD(isnan(BasalFD) | (BasalFD < THRESHOLD)) = [];
ApicalFD(isnan(ApicalFD) | (ApicalFD < THRESHOLD)) = [];

fdStats.usedSlices = numel(GlobalFD);

if isempty(GlobalFD)
  fdStats.globalFD = NaN;
else
  fdStats.globalFD = nanmean(GlobalFD);
end

if isempty(BasalFD)
  fdStats.meanBasalFD = NaN;
  fdStats.maxBasalFD  = NaN;
else
  fdStats.meanBasalFD = nanmean(BasalFD);
  fdStats.maxBasalFD  = nanmax(BasalFD);
end

if isempty(ApicalFD)
  fdStats.meanApicalFD = NaN;
  fdStats.maxApicalFD  = NaN;
else
  fdStats.meanApicalFD = nanmean(ApicalFD);
  fdStats.maxApicalFD  = nanmax(ApicalFD);
end

end



      
      
      
      


