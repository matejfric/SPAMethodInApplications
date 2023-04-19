function [Pi,classes] = myonehotencode(labels)
%MYONEHOTENCODE
arguments
    labels (:,1) double       
end
%OUT: Pi (:,:) double   ... M x T matrix
%     classes           ... unique classification classes

classes = unique(labels,'stable'); % do not sort
T = length(labels);
M = length(classes);
Pi = zeros(M,T);

for m=1:M
    Pi(m, labels == classes(m)) = 1;
end


end

