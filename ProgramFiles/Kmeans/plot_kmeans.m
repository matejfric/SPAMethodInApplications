function [] = plot_kmeans(T, Gamma, centroids, k)
%PLOT_KMEANS

figure
hold on
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
%title('K-means','FontSize', 14);
grid on
grid minor

%colors = 'bgmyc'; 1.0000    0.6000    0.8000 [0.8500 0.3250 0.0980]
colors = [[0 0.4470 0.7410]; [0.4660 0.6740 0.1880]; [0.3010 0.7450 0.9330]; [0.9290 0.6940 0.1250]; [1 0.6 0.8]];
markers = ["square", "diamond", "^", "pentagram", "hexagram"]; %'x+*ph.'
clusters = string.empty;

%% draw clusters
for i = 1:k
        ids = Gamma(i,:)==1;
        scatter(T(ids,1), T(ids,2), 30,...
            'MarkerFaceColor',colors(i,:), 'Marker', markers(i),...
            'MarkerEdgeColor',colors(i,:), 'LineWidth',1,...
            'MarkerFaceAlpha', 0.5)
        %clusters(i)= 'cluster ' + string(i);
        clusters(i)= string(i);
end
%clusters(k+1)='centroids';

%% draw centroids
scatter(centroids(:,1),centroids(:,2),40 ,...
    'MarkerEdgeColor','r','MarkerFaceColor','r',...
    'MarkerFaceAlpha', 1)

lgd = legend(clusters,'Location','Southwest');
title(lgd,'Cluster ID')

end

