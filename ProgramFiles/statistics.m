function [stats] = statistics(labels, ground_truth)
%STATISTICS Summary of this function goes here

if isnumeric(labels) 
    labels = round(labels); 
end
if isnumeric(ground_truth) 
    ground_truth = round(ground_truth); 
end

% Confusion matrix
C = confusionmat(labels, ground_truth);

TN = C(1,1);
FP = C(1,2);
FN = C(2,1);
TP = C(2,2);

stats.fp = FP;
stats.fn = FN;

%confusionchart(C);

% PRECISION
stats.precision = TP / (TP + FP);

% RECALL
if (TP + FN) == 0
    stats.recall = 0;
else
    stats.recall = TP / (TP + FN);
end

% F1SCORE
if (stats.precision + stats.recall) == 0
    stats.f1score = 0;
else
    stats.f1score = 2 * (stats.precision * stats.recall) /...
                    (stats.precision + stats.recall); 
end
                
% ACCURACY
stats.accuracy = (TP + TN) / ( TP + TN + FP + FN );

% MEAN ABSOLUTE ERROR (MAE)
% https://en.wikipedia.org/wiki/Mean_absolute_error
if size(labels, 2) > size(labels, 1)
    labels = labels'; 
end
if ~isnumeric(labels) 
    labels = onehotencode(labels,2); 
end
if size(ground_truth, 2) > size(ground_truth, 1)
    ground_truth = ground_truth'; 
end
if ~isnumeric(ground_truth) 
    ground_truth = onehotencode(ground_truth,2); 
end
stats.mae = sum(abs(labels(:,1) - ground_truth(:,1))) / size(ground_truth,1);

% MEAN SQUARED ERROR (MSE)
% https://en.wikipedia.org/wiki/Mean_squared_error
stats.mse = sum((labels(:,1) - ground_truth(:,1)).^2) / size(ground_truth,1);

end

