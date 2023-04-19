function [labels] = myonehotdecode(Pi,classes)
%MYONEHOTENCODE 
arguments
    Pi (:,:) double       % M x T matrix
    classes (:,1) double  % uniques classes
end

[M,T] = size(Pi);

labels = zeros(T,1);

PiY_int = round(Pi);

for m=1:M
    labels(PiY_int(m,:) == 1) = classes(m);
end


end

