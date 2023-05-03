function [] = plot_initialization(X, C)
%PLOT_INITIALIZATION Plot initial centroids for the k-means algorithm
% X ... D, T
% C ... D, K

figure
hold on
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
%title('k-means initialization','FontSize', 14);
grid on
grid minor

K = size(C,2);

color_blue = [0 0.4470 0.7410];

%% draw points
scatter(X(1,:), X(2,:), 30,'Marker', 'square', ...
    'MarkerEdgeColor',color_blue,'MarkerFaceColor',color_blue,...
    'MarkerFaceAlpha', 0.5, 'LineWidth', 1)

%% draw centroids
scatter(C(1,:),C(2,:),40 ,'MarkerEdgeColor','r','MarkerFaceColor','r')

end

