function [] = score_plot(mytitle, range, ...
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
if max(range) <= 1
    xlabel('$\alpha$','Interpreter','latex')
else
    xlabel('K')
end
ylabel('Score')
xticks(range)
if max(range) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy',  'Location', 'southoutside')
title('Training Phase')
grid on;
grid minor;
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
if max(range) <= 1
    xlabel('$\alpha$','Interpreter','latex')
else
    xlabel('K')
end
ylabel('Score')
xticks(range)
if max(range) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy',  'Location', 'southoutside')
title('Testing Phase')
sgtitle(mytitle)
grid on;
grid minor;
hold off

end

function [] = myplot(x, y)

plot(x, y, 'o-' , 'LineWidth', 2);

end

