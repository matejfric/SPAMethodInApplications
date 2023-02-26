function [stats] = adamar_validate_fisher_iris(Lambda, C, Y, PiY_true, classes)
%ADAMAR_PREDICT Make a prediction based on ADAMAR model
%   Lambda...transion matrix
%   C........centroids
%   K........number of clusters
arguments
    Lambda, C, Y, PiY_true, classes
end

n = size(PiY_true, 1);

for i = 1:n
    [Gamma] = compute_Gamma_kmeans(C,Y'); % Gamma reconstruct
    
    %PiY = round(Lambda*Gamma)'; % round => binary matrix
    PiY = (Lambda*Gamma)'; % round => binary matrix

    % Statistics
    labels = onehotdecode(PiY,classes,2);
    ground_truth = onehotdecode(PiY_true',classes,2);
    [stats] = statistics_multiclass(labels, ground_truth);
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


