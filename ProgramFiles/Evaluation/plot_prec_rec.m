function [AUCpr] = plot_prec_rec(y_true,y_pred)
%PLOT_PREC_REC Plot Precision-Recall curve and compute AUC score
% inspired by: https://www.quora.com/How-can-I-plot-a-precision-recall-curve-in-MATLAB

[Xpr,Ypr,Tpr,AUCpr] = perfcurve(y_true,y_pred, 1,...
    'xCrit', 'reca', 'yCrit', 'prec'); 

% Plot Precision-Recall curve
figure
plot(Xpr,Ypr,'DisplayName', 'Precision-Recall curve', 'Linewidth', 2)

xlabel('Recall', 'FontSize',13, 'Interpreter', 'latex')
ylabel('Precision', 'FontSize',13, 'Interpreter', 'latex')
title(['Precision-Recall Curve (AUC = ' num2str(AUCpr,3) ')'], 'Interpreter', 'latex', 'FontSize',15)
legend('show', 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize',14)

end

