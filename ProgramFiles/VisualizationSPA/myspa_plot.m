function [] = myspa_plot(X, Gamma, S, K)
%SPA_PLOT Plot clustering provided by SPA
%
% Gamma ... K x T
% S     ... D x K
% X     ... D x T
% K     ... scalar (number of clusters)

figure
hold on
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
grid on
grid minor

colors = [[0 0.4470 0.7410]; [0.4660 0.6740 0.1880]; [0.3010 0.7450 0.9330]; [0.9290 0.6940 0.1250]; [1 0.6 0.8]];
markers = ["square", "diamond", "^", "pentagram", "hexagram"]; %'x+*ph.'
clusters = string.empty;

Gamma = binarize_Gamma(Gamma);

%% Draw clusters
for i = 1:K
        ids = Gamma(i,:)==1;
        scatter(X(1,ids), X(2,ids), 30,...
            'MarkerFaceColor',colors(i,:), 'Marker', markers(i),...
            'MarkerEdgeColor',colors(i,:), 'LineWidth',1,...
            'MarkerFaceAlpha', 0.5)
        clusters(i)= string(i);
end

%% Draw centroids
scatter(S(1,:),S(2,:),40 ,...
    'MarkerEdgeColor','r','MarkerFaceColor','r',...
    'MarkerFaceAlpha', 1)

lgd = legend(clusters,'Location','Southwest');
title(lgd,'Cluster ID')

end

function Gamma = binarize_Gamma(Gamma)
%BINARIZE_GAMMA
% Gamma ... K x T

    [~,idx] = max(Gamma);
    Gamma = onehotencode(categorical(idx),1);

end
