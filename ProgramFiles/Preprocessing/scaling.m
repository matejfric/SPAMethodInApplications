function [X, ca_Y, method, SCALE] = scaling(...
    X, ca_Y, method, methodType, colmin, colmax)
% SCALING Scaling the training matrix and test dataset. This implementation
%         is incorrect. It is recommended to use scaling2 instead.

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
    cols = colmax > 1 | colmin < 0; % Select columns to be scaled - incorrect
    X(:,cols) = l + ((X(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
    %X = l + ((X-colmin)./(colmax-colmin)).*(u-l); 
    n = numel(ca_Y);
    for i = 1:n
        Y = ca_Y{i}.X;
        Y(:,cols) = l + ((Y(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
        %Y(:,1:end-1) = l + ((Y(:,1:end-1)-colmin)./(colmax-colmin)).*(u-l);
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

