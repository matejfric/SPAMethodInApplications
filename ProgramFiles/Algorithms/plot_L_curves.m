function [] = plot_L_curves(Ls, L1s, L2s, alphas, K, mytitle)
%PLOT_L_CURVES

arguments
    Ls, L1s, L2s, alphas, K,
    mytitle = sprintf('K=%d', K)
end

[w, h] = get_screen_resolution();
figure('Renderer', 'painters', 'Position', [w-w*0.8-(w*(1-0.8))/2 h/4 w*0.8 h/2])


tiledlayout(1,5)
% log-log L-curve
nexttile
hold on
title('log-log L-curve')
plot(L1s, L2s,'r-o');
%set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log')
grid minor
xlabel('$L1$','Interpreter','latex')
ylabel('$L2$','Interpreter','latex')
%text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
%text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
for i = 1:numel(L1s)
    text(L1s(i),L2s(i),['$\alpha = ' num2str(alphas(i)) '$'],'Interpreter','latex')
end
hold off

nexttile
hold on
title('lin-lin L-curve')
plot(L1s, L2s,'r-o');
grid minor
xlabel('$L1$','Interpreter','latex')
ylabel('$L2$','Interpreter','latex')
%text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
%text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
for i = 1:numel(L1s)
    text(L1s(i),L2s(i),['$\alpha = ' num2str(alphas(i)) '$'],'Interpreter','latex')
end
hold off

% L, L1, L2
%subplot(1,4,2)
nexttile
hold on
plot(alphas,Ls,'r*-')
grid minor
title('L')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L$','Interpreter','latex')

%subplot(1,4,3)
nexttile
hold on
plot(alphas,L1s,'b*-')
grid minor
title('L1')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L_1$','Interpreter','latex')

%subplot(1,4,4)
nexttile
hold on
plot(alphas,L2s,'m*-')
grid minor
title('L2')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L_2$','Interpreter','latex')
hold off

sgtitle(mytitle) 

end



