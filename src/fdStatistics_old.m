function fdStats = fdStatistics_old(fdData,discardMode)
% Copyright (c) 2017 Cai Jiashen

fdStats.evalSlices = numel(fdData);

[quot, rem] = quorem(sym(fdStats.evalSlices),2);

if discardMode == false
    first = 1;
    last = fdStats.evalSlices;
    fdStats.usedSlices = fdStats.evalSlices;
elseif discardMode == true
    first = 2;
    last = fdStats.evalSlices - 1;
    fdStats.usedSlices = fdStats.evalSlices - 2;
    quot = quot - 1;
end

if isempty(fdData)
  fdStats.globalFD = NaN;
else
  fdStats.globalFD = nanmean(fdData);
end

basalSlices = [first:first+quot-1];
apicalSlices = [first+quot+rem:last];

if isempty(fdData(apicalSlices))
  fdStats.meanApicalFD = NaN;
  fdStats.maxApicalFD  = NaN;
else
  fdStats.meanApicalFD = nanmean(fdData(apicalSlices));
  fdStats.maxApicalFD  = nanmax(fdData(apicalSlices));  
end    

if isempty(fdData(basalSlices))
  fdStats.meanBasalFD = NaN;
  fdStats.maxBasalFD  = NaN;
else
  fdStats.meanBasalFD = nanmean(fdData(basalSlices));
  fdStats.maxBasalFD  = nanmax(fdData(basalSlices));
end
