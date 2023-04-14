function [best_alpha, best_K, best_score] = grid_search(stats_val,alphas,Ks)
%GRID_SEARCH 

scores = zeros(size(stats_val));
for j=1:size(scores,2)
    scores(:,j) = [stats_val(:,j).f1score]';
end

[XX,YY] = meshgrid(alphas,Ks);

figure
surf(XX,YY,scores','FaceColor','interp','FaceAlpha',0.9)
colorbar;
colormap jet;
[i,j,best_score] = max_in_matrix(scores);
xlabel('$\alpha$','interpreter','latex')
ylabel('$K$','interpreter','latex')
zlabel('$F_1$-score','interpreter','latex') 
hold on
best_alpha = alphas(i);
best_K = Ks(j);
scatter3(best_alpha,best_K,best_score,100,'^k','filled');
text(best_alpha, best_K, best_score,...
    sprintf('($\\alpha$=%.4f, $K$=%d)', best_alpha, best_K),...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',...
    'interpreter','latex','FontSize',14);

end

