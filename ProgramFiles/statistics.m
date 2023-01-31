function [stats] = statistics(labels, ground_truth)
%STATISTICS Summary of this function goes here

labels = round(labels);

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


end

