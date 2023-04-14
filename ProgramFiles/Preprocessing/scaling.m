function [X, ca_Y, method, SCALE] = scaling(...
    X, ca_Y, method, methodType, colmin, colmax)
% SCALING Scale training matrix and testing dataset.
%--------------------------------------------------------------------------
% Results for SVM_X, training on X10 (first 10 images in 'Dataset'),
% testing on '68.jpg' (pipe wrench). List of different methods follows:
%--------------------------------------------------------------------------
% 'minmax' ~0.80 ...scale cols with max(col) > 1 to [-1,1]
% 'none' ~0.76
% 'center' ~0.74
% 'zscore' ~0.67
% 'range' [0,1] ~0.71
% 'scale' ~0.63
% 'medianiqr' ~0.57
% 'norm' ~0.00
%--------------------------------------------------------------------------
% SVM_X 'Dataset2'
% 'minmax' ~0.40
% 'none' ~0.35
%--------------------------------------------------------------------------
arguments
    X, ca_Y, method, methodType = [], colmin = [], colmax = []
end

if strcmp(method, 'none')
    return;
elseif strcmp(method, 'minmax')
    if isempty(colmin) || isempty(colmax)
        colmin = min(X); % a
        colmax = max(X); % b
    end
    u = 1;
    l = 0;
    cols = colmax > 1; % Select columns to be scaled
    X(:,cols) = l + ((X(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
    n = numel(ca_Y);
    for i = 1:n
        Y = ca_Y{i}.X;
        Y(:,cols) = l + ((Y(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
        ca_Y{i}.X = Y;
    end
else
    if isempty(methodType)
        X = normalize(X, method);
    else
        X = normalize(X, method, methodType);
    end
    n = numel(ca_Y);
    for i = 1:n
        if isempty(methodType)
            ca_Y{i}.X(:,1:end-1) = normalize(ca_Y{i}.X(:,1:end-1), method);
        else
            ca_Y{i}.X(:,1:end-1) = normalize(ca_Y{i}.X(:,1:end-1), method, methodType);
        end
    end
end

SCALE.colmin = colmin;
SCALE.colmax = colmax;

end

