function [PiY] = myonehotencode(labels,featureDim)
%MYONEHOTENCODE Summary of this function goes here
%   Detailed explanation goes here

categ = unique(labels);
T = length(labels);
M = length(categ);
PiY = zeros(T,M);

for m=1:M
    PiY(labels == categ(m),m) = 1;
end


end

