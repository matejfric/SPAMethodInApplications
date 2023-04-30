function [stats] = statistics(labels, ground_truth)
%STATISTICS

try
    [~,~,~,AUC] = perfcurve(ground_truth,labels,1);
    stats.auc = AUC;
    
    [~,~,~,PRAUC] = perfcurve(ground_truth,labels,1,...
    'xCrit', 'reca', 'yCrit', 'prec'); 
    stats.prauc = PRAUC;
catch ME
    warning("%s\n%s\n", ME.identifier, ME.message);
    stats.auc = NaN;
    stats.prauc = NaN;
end 

if isnumeric(labels) 
    labels = round(labels);
end
if isnumeric(ground_truth) 
    ground_truth = round(ground_truth); 
end

% Confusion matrix
CM = confusionmat(labels, ground_truth);

TN = CM(1,1);
FP = CM(1,2);
FN = CM(2,1);
TP = CM(2,2);

stats.CM = CM;
stats.tp = TP;
stats.tn = TN;
stats.fp = FP;
stats.fn = FN;

%confusionchart(C);

% PRECISION
if (TP + FP) == 0
    stats.precision = 0;
else
    stats.precision = TP / (TP + FP);
end

% RECALL
if (TP + FN) == 0
    stats.recall = 0;
else
    stats.recall = TP / (TP + FN);
end

% F1SCORE
if (TP+FP+FN) == 0
    stats.f1score = 1;
else
    stats.f1score = 2*TP / (2*TP + FP + FN);
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

% Calculate Matthews Correlation Coefficient (MCC)
num = (TP*TN - FP*FN);
denom = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if denom == 0
    mcc = 0; % TODO: this can be defined better than as zero
    warning("MCC undefined.");
else
    mcc = num / denom; 
end

stats.mcc = mcc;
stats.nmcc = (mcc+1)/2;

end

