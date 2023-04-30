function [best_alpha, best_K, best_mcc] = grid_search_mcc(stats_val,alphas,Ks)
%GRID_SEARCH 

% One dimensional grid case 
if ~(sum(size(stats_val) >= [2,2]) >= 2)
    if length(Ks) > length(alphas)
        [best_mcc, k] = max([stats_val.mcc]);
        best_K = Ks(k);
        best_alpha = alphas;
        
        figure
        plot(Ks, [stats_val.mcc], 'o-', 'LineWidth', 2)
        hold on
        plot(best_K, best_epsilon, 'Marker', '^', 'MarkerSize', 15)
        xlabel('$K$','interpreter','latex','FontSize',14)
        ylabel('MCC','interpreter','latex','FontSize',14)
    else
        [best_mcc, a] = max([stats_val.mcc]);
        best_alpha = alphas(a);
        best_K = Ks;
        
        figure
        plot(alphas, [stats_val.mcc], 'o-', 'LineWidth', 2)
        hold on
        plot(best_alpha, best_mcc, 'Marker', '^', 'MarkerSize', 15)
        xlabel('$\alpha$','interpreter','latex','FontSize',14)
        ylabel('MCC','interpreter','latex','FontSize',14)
    end
    return
end


% Two dimensional grid case 
scores = zeros(size(stats_val));
for j=1:size(scores,2)
    scores(:,j) = [stats_val(:,j).mcc]';
end

[XX,YY] = meshgrid(alphas,Ks);

figure
surfc(XX,YY,scores','FaceColor','interp','FaceAlpha',0.9)
colorbar;
colormap jet;
[i,j,best_mcc] = max_in_matrix(scores);
xlabel('$\alpha$','interpreter','latex')
ylabel('$K$','interpreter','latex')
zlabel('MCC','interpreter','latex') 
hold on
best_alpha = alphas(i);
best_K = Ks(j);
scatter3(best_alpha,best_K,best_mcc,100,'^k','filled');
text(best_alpha, best_K, best_mcc,...
    sprintf('($\\alpha$=%.4f, $K$=%d)', best_alpha, best_K),...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom',...
    'interpreter','latex','FontSize',14);

end

