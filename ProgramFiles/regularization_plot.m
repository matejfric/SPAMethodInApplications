function [] = regularization_plot(mytitle, range, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)
%SCORE_PLOT 

% Training
figure
subplot(1,2,1)
set(gca,'DefaultLineLineWidth',2)
myplot(range,lprecision) 
hold on 
myplot(range,lrecall)
hold on 
myplot(range,lf1score)
hold on 
myplot(range,laccuracy)
xlabel('alpha')
ylabel('Score')
xticks(range)
set(gca, 'XScale', 'log')
legend('Precision','Recall', 'F1-score', 'Accuracy', 'Location','southeast')
title('Training Phase')
hold off

% Testing
subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
myplot(range,tprecision) 
hold on 
myplot(range,trecall)
hold on 
myplot(range,tf1score)
hold on 
myplot(range,taccuracy)
xlabel('alpha')
ylabel('Score')
xticks(range)
set(gca, 'XScale', 'log')
legend('Precision','Recall', 'F1-score', 'Accuracy', 'Location','southeast')
title('Testing Phase')
sgtitle(mytitle)
hold off

end

function [] = myplot(x, y)

plot(x, y, 'o-' , 'LineWidth', 2);

end

