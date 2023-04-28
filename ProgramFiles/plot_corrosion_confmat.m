function [] = plot_corrosion_confmat(stats)
%MYCONFMAT

CM = [stats.tp  stats.fn
      stats.fp  stats.tn];
 
figure;
cm = confusionchart(CM, categorical(["corrosion","not_corrosion"]));
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
%cm.YLabel = 'Actual class';
cm.FontSize = 14;

end

