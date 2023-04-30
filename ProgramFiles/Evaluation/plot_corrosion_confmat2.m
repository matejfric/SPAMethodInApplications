function [] = plot_corrosion_confmat2(y_true,y_pred)
%MYCONFMAT

y_true = ~y_true;
y_pred = ~round(y_pred);

figure;
cm = plotconfusion(y_true,y_pred);
title("")
fh = gcf; % access the figure handle for the confusion matrix plot
ax = fh.Children(2); % access the corresponding axes handle
ax.XTickLabel{1} = '\texttt{corrosion}';
ax.YTickLabel{1} = '\texttt{corrosion}';
ax.XTickLabel{2} = '\texttt{not\_corrosion}'; 
ax.YTickLabel{2} = '\texttt{not\_corrosion}';
ax.TickLabelInterpreter = 'latex';
ax.XLabel.String = 'Predicted class';
ax.XLabel.FontWeight = 'normal';
ax.XLabel.Interpreter = 'latex';
%ax.XLabel.Position(2)=-20+ax.XLabel.Position(2);
ax.YLabel.String = 'Actual class'; 
ax.YLabel.FontWeight = 'normal';
ax.YLabel.Interpreter = 'latex';
%ax.YLabel.Position(1)=+20+ax.YLabel.Position(1);
set(gca, 'FontSize', 15);

end

