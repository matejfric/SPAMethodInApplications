function [Pi,classes] = myonehotencode(labels, classes)
%MYONEHOTENCODE
arguments
    labels (:,1) double  
    classes = []
end
%OUT: Pi (:,:) double   ... M x T matrix
%     classes           ... unique classification classes

if isempty(classes)
    classes = unique(labels,'stable'); % do not sort
    classes = sort(classes,'descend');
end

T = length(labels);
M = length(classes);
Pi = zeros(M,T);

for m=1:M
    Pi(m, labels == classes(m)) = 1;
end


end

