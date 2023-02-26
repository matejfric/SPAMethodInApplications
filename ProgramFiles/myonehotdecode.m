function [labels] = myonehotdecode(PiY,classes,featureDim)
%MYONEHOTENCODE Summary of this function goes here
%   Detailed explanation goes here

[T,M] = size(PiY);

labels = zeros(T,1);

PiY_int = round(PiY);

for m=1:M
    labels(PiY_int(:,m) == 1) = m;
end


end

