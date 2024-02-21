% bxct Box-counting of binary image for computation of fractal dimension
% Copyright (c) 2016 Cai Jiashen

function [fd, bcFig] = bxct(I)

%Padding image to have equal dimensions
s = regionprops(I, 'BoundingBox');

if (s(1).BoundingBox(3)*s(1).BoundingBox(4) > 0.25*size(I,1)*size(I,2))
    I = imcrop(I, s(1).BoundingBox);
end


[height, width]=size(I);

if height > width
    padWidth = height-width;
    padHeight = 0;
    dimen = height;
else
    padHeight = width-height;
    padWidth = 0;
    dimen = width;
end

padI = padarray(I, [mod(padHeight,2), mod(padWidth,2)],'pre');
padI = padarray(padI,[floor(padHeight/2), floor(padWidth/2)]);

%imshow (padI);

tic;
[nBoxA, boxSizeA] = initGrid(padI,dimen);
[nBoxB, boxSizeB] = initGrid(flip(padI,1),dimen);
[nBoxC, boxSizeC] = initGrid(flip(padI,2),dimen);
[nBoxD, boxSizeD] = initGrid(flip(flip(padI,1),2),dimen);

toc

nBox = min([nBoxA; nBoxB; nBoxC; nBoxD]); % Maximal efficient covering
boxSize = boxSizeA;

totalBoxSizes = numel(boxSize);

p = polyfit(log(boxSize),log(nBox),1);
fd = -p(1);

% Box Count Plot
bcFig = figure('Name','BoxCount Plot','NumberTitle','off','Visible','off');
pause(0.01);
set(0, 'CurrentFigure', bcFig);
loglog(boxSizeA,nBoxA,'s-');
xlabel('r, box size (pixels)'); ylabel('n(r), box count');


x1=linspace(2,0.45*dimen);
y1=exp(polyval(p,log(x1)));
hold on;
loglog(x1,y1);

legend('Box Count','Regression');

    function [nBox, boxSize]= initGrid(I,dimen)
        
        startBoxSize = floor(0.45*dimen);
        curBoxSize = startBoxSize;
        
        nBox = zeros(1,(startBoxSize-2+1));
        boxSize= [startBoxSize:-1:2];
        for sizeCount = 1:(startBoxSize-2+1)
            curBoxSize = boxSize(sizeCount);
            
            for macroY = 1:ceil(dimen/curBoxSize)
                for macroX = 1:ceil(dimen/curBoxSize)
                    boxYinit = (macroY-1)*curBoxSize+1;
                    boxXinit = (macroX-1)*curBoxSize+1;
                    boxYend = min(macroY*curBoxSize,dimen);
                    boxXend = min(macroX*curBoxSize,dimen);
                    
                    boxFound = false;
                    for curY = boxYinit:boxYend
                        for curX = boxXinit:boxXend
                            if I(curY,curX)
                                boxFound = true;
                                nBox(sizeCount) = nBox(sizeCount) + 1;
                                break;
                            end
                        end
                        
                        if boxFound == true
                            break;
                        end
                    end
                    
%                     if ~isempty(find(I(boxYinit:boxYend,boxXinit:boxXend),1,'first'))
%                         nBox(sizeCount) = nBox(sizeCount)+1;
%                     end
                    
%                     if sum(sum(I(boxYinit:boxYend,boxXinit:boxXend))) > 0
%                         nBox(sizeCount) = nBox(sizeCount)+1;
%                     end
                    
                end
            end
        end    
    end
end