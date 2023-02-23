function [stats] = statistics_multiclass(labels, ground_truth)
%f1score Summary of this function goes here
%   Detailed explanation goes here
% CM = [ 50 3 0 0;
%        26 8 0 1;
%        20 2 4 0;
%        12 0 0 1 ]

labels = round(labels);
ground_truth = round(ground_truth);

% Confusion matrix
CM = confusionmat(labels, ground_truth);
%figure; confusionchart(CM);
CM = CM';

TPs = diag(CM);
FPs = sum(CM,2) - TPs;
FNs = sum(CM,1)' - TPs;
precision = TPs./sum(CM,2);
recall = TPs./sum(CM,1)';

f1_scores = 2*(precision.*recall)./(precision+recall);
if sum(isnan(f1_scores)) > 0
    f1_scores(isnan(f1_scores)) = 0;
end
accuracy_scores = (TPs)./(TPs+FPs+FNs);

stats.f1score = mean(f1_scores);
stats.precision = mean(precision);
stats.recall = mean(recall);
stats.accuracy = mean(accuracy_scores);

end

