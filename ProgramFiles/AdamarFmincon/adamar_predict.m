function [stats] = adamar_predict(Lambda, C, K, a, b, images, descriptors)
%ADAMAR_PREDICT Make a prediction based on ADAMAR model
%   Lambda...transion matrix
%   C........centroids
%   K........number of clusters
%   a........colmin of X
%   b........colmax of X

if(isempty(images))
    %image_number = 137; % 137...hřebík
    images = 68;
end

n = numel(images);

for i = 1:n
    % Load image and get descriptors
    ca = load_images(images(i));
    Y = get_descriptors(ca, descriptors);
    
    % % Normalization
    %Y(:,1:end-1) = normalize(Y(:,1:end-1));
    
    % MinMaxScaling
    if ~isempty(a) && ~isempty(b)
        % Selective MinMaxScaling [-1,1]
        l = -1;
        u = 1;
        cols = b > 1; % Select columns to be scaled
        Y(:,cols) = l + ...
            ((Y(:,cols)-a(cols))./(b(cols)-a(cols))).*(u-l);
    end
    
    % K-means (one step)
    GammaY = compute_Gamma_kmeans(C,Y(:,1:end-1)');
    
    PiY = round(Lambda*GammaY)'; % round => binary matrix
    %PiY = (Lambda*GammaY)'; % round => binary matrix
    
    % Statistics
    stats = statistics(PiY(:,1), Y(:,end));
    if i == 1
        stats_avg = stats;
    else
        stats_avg.fp = stats_avg.fp + stats.fp;
        stats_avg.fn = stats_avg.fn + stats.fn;
        stats_avg.precision = stats_avg.precision + stats.precision;
        stats_avg.recall = stats_avg.recall + stats.recall;
        stats_avg.f1score = stats_avg.f1score + stats.f1score;
        stats_avg.accuracy = stats_avg.accuracy + stats.accuracy;
    end
end

stats_avg.precision = stats_avg.precision / n;
stats_avg.recall = stats_avg.recall / n;
stats_avg.f1score = stats_avg.f1score / n;
stats_avg.accuracy = stats_avg.accuracy / n;

stats = stats_avg;

visualize(ca(1,1), ca(1,2), Y(:,end), PiY(:,1), sprintf('Adamar K-means, K=%d', K));
pause(1)

end

