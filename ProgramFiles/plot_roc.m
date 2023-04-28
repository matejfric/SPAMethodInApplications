function [TPR,FPR,T,AUC] = plot_roc(y_true, y_pred)
%PLOT_ROC Plot ROC curve

[TPR,FPR,T,AUC,OPTROCPT] = perfcurve(y_true,y_pred,1);

% Plot ROC curve
figure
plot(TPR,FPR, 'DisplayName', 'ROC curve', 'Linewidth', 2)
hold on
plot(OPTROCPT(1),OPTROCPT(2),...
    'ro', 'DisplayName', 'Optimal ROC operating point', 'Linewidth', 2)
xlabel('False positive rate (FPR)', 'FontSize',13, 'Interpreter', 'latex')
ylabel('True positive rate (TPR)', 'FontSize',13, 'Interpreter', 'latex')
title(['ROC curve (AUC = ' num2str(AUC,3) ')'], 'Interpreter', 'latex', 'FontSize',15)
legend('show', 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize',14)

end

