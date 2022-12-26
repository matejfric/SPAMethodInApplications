function [stats] = adamar_predict_mat(Lambda, C, K, a, b, ca_Y)
%ADAMAR_PREDICT Make a prediction based on ADAMAR model
%   Lambda...transion matrix
%   C........centroids
%   K........number of clusters
%   a........colmin of X
%   b........colmax of X
%   ca_Y.....matrices of descriptors in cell array

n = numel(ca_Y);

for i = 1:n
    Y = ca_Y;
    %Y = ca_Y{i}.X;
    
    % % Normalization
    %Y(:,1:end-1) = normalize(Y(:,1:end-1));
    
    % MinMaxScaling
    if ~isempty(a) && ~isempty(b)
        Y = rescale(Y,'InputMin',a,'InputMax',b); 
    end

    % K-means
    dist_from_C = zeros(K, size(Y,1));
    for k=1:K
        dist_from_C(k,:) = sum((Y(:,1:end-1)' - C(:,k)).^2,1);
    end
    [~,idxY] = min(dist_from_C);
    GammaY = zeros(K,length(idxY));
    for k = 1:K
       GammaY(k,idxY==k) = 1; 
    end
    PiY = round(Lambda*GammaY)'; % round => binary matrix

    % Statistics
    stats = statistics(PiY(:,1), Y(:,end));
    if i == 1
        stats_avg = stats;
    else
        stats_avg.precision = stats.precision + stats_avg.precision;
        stats_avg.recall = stats.recall + stats_avg.recall;
        stats_avg.f1score = stats.f1score + stats_avg.f1score;
        stats_avg.accuracy = stats.accuracy + stats_avg.accuracy;
    end
end

stats_avg.precision = stats_avg.precision / n;
stats_avg.recall = stats_avg.recall / n;
stats_avg.f1score = stats_avg.f1score / n;
stats_avg.accuracy = stats_avg.accuracy / n;

stats = stats_avg;

end

