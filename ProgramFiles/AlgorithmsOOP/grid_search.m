function [best_alpha, best_K, best_score] = grid_search(stats_val,alphas,Ks)
%GRID_SEARCH 

% One dimensional grid case 
if ~(sum(size(stats_val) >= [2,2]) >= 2)
    if length(Ks) > length(alphas)
        [best_score, k] = max([stats_val.f1score]);
        best_K = Ks(k);
        best_alpha = alphas;
        
        figure
        plot(Ks, [stats_val.f1score], 'o-', 'LineWidth', 2)
        hold on
        plot(best_K, best_score, 'Marker', '^', 'MarkerSize', 15)
        xlabel('$K$','interpreter','latex','FontSize',14)
        ylabel('$F_1$-score','interpreter','latex','FontSize',14)
    else
        [best_score, a] = max([stats_val.f1score]);
        best_alpha = alphas(a);
        best_K = Ks;
        
        figure
        plot(alphas, [stats_val.f1score], 'o-', 'LineWidth', 2)
        hold on
        plot(best_alpha, best_score, 'Marker', '^', 'MarkerSize', 15)
        xlabel('$\alpha$','interpreter','latex','FontSize',14)
        ylabel('$F_1$-score','interpreter','latex','FontSize',14)
    end
    return
end


% Two dimensional grid case 
scores = zeros(size(stats_val));
for j=1:size(scores,2)
    scores(:,j) = [stats_val(:,j).f1score]';
end

[XX,YY] = meshgrid(alphas,Ks);

figure
surfc(XX,YY,scores','FaceColor','interp','FaceAlpha',0.9)
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

