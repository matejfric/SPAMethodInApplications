function [stats] = compute_training_stats(PiY, PiX)
%COMPUTE_TRAINING_STATS
if size(PiY, 1) > 2 % multi-class classification
    c = size(PiY, 1); % number of classes
    PiX = round(PiX);
    R = PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1);
    r = randi([1 c],1,size(R,2));
    PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1) = bsxfun(@eq, r(:), 1:c)';
    [prediction, ~] = find(PiX);
    [ground_truth, ~] = find(round(PiY));
    if length(prediction) ~= length(ground_truth)
        keyboard
    end
    stats = statistics_multiclass(prediction, ground_truth);
    
else % binary classification
    ground_truth = PiY(1,:);
    stats = statistics(PiX(1,:), ground_truth);
%   learningErrors(i) = sum(abs(PiX(:,1) - ground_truth')) / length(ground_truth);
end

end

