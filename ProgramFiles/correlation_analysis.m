function [X, ca_Y] = correlation_analysis(X, ca_Y, display)
%CORRELATION_ANALYSIS
arguments
    X (:,:) double
    ca_Y
    display = false
end

E = corr(X);
if display; plot_correlation(E); end

E = E - eye(size(E,1));
remove=any( E > 0.95 | E < -0.95);
E=E(~remove,~remove);
if display; plot_correlation(E); end

% Remove highly correlated columns from X 
remove(end) = 0;
X(:,remove) = [];

% Remove highly correlated columns from ca_Y 
n = numel(ca_Y);
for i = 1:n
    ca_Y{i}.X(:,remove) = [];
end

