function [] = fmincon_score_plot(mytitle, iterations, images, ...
    lprecision, lrecall, lf1score, laccuracy,...
    tprecision, trecall, tf1score, taccuracy)
%SCORE_PLOT 

hold off

% Training
range = iterations;
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
xlabel('Iteration')
ylabel('Score')
xticks(range)
if max(range) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy', 'Location','southeast')
title('Training Phase')
hold off

% Testing
range = images;
subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
myplot(range,tprecision) 
hold on 
myplot(range,trecall)
hold on 
myplot(range,tf1score)
hold on 
myplot(range,taccuracy)
xlabel('Image')
ylabel('Score')
xticks(range)
if max(range) >= 100
    set(gca, 'XScale', 'log')
end
legend('Precision','Recall', 'F1-score', 'Accuracy', 'Location','southeast')
title('Testing Phase')
sgtitle(mytitle)
hold off

end

function [] = myplot(x, y)

plot(x, y, 'o-' , 'LineWidth', 2);

end

