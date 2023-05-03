function [] = myspa_plot2(X, Gamma, S, K)
%SPA_PLOT Plot clustering provided by SPA
%
% Gamma ... K x T
% S     ... D x K
% X     ... D x T
% K     ... scalar (number of clusters)

% Draw clusters
for i = 1:K
    figure
hold on
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
grid on
grid minor

XS = [X,S];
GammaS = [Gamma,eye(K)];

x = XS(1,:);
y = XS(2,:);
z = GammaS(i,:);
xlin = linspace(min(x), max(x), 100);
ylin = linspace(min(y), max(y), 100);
[XX,YY] = meshgrid(xlin, ylin);
ZZ = griddata(x,y,z,XX,YY,'natural');
mesh(XX,YY,ZZ)


% Draw centroids
scatter(S(1,:),S(2,:),40 ,...
    'MarkerEdgeColor','r','MarkerFaceColor','r',...
    'MarkerFaceAlpha', 1)

end


end
