function [E] = plot_correlation(E)
%PLOT_CORRELATION

figure;
imagesc(E); % Display correlation matrix as an image
set(gca, 'XTick', 1:size(E,2)); % center x-axis ticks on bins
set(gca, 'YTick', 1:size(E,2)); % center y-axis ticks on bins
% set(gca, 'XTickLabel', yourlabelnames); % set x-axis labels
% set(gca, 'YTickLabel', yourlabelnames); % set y-axis labels
title('Correlation', 'FontSize', 10); % set title
colormap('jet'); % Choose jet or any other color scheme
colorbar; %

end

