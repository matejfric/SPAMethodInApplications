function [] = plot_L_curves(Ls, L1s, L2s, Ks, alphas)
%PLOT_L_CURVES

% L-curve
for k=1:length(Ks)
    figure
    hold on
    title(sprintf('akmeans, K=%d', Ks))
    plot(L1s(:,k), L2s(:,k),'r-o');
    %text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    %text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    for i = 1:numel(L1s(:,k))
        text(L1s(i),L2s(i),['$\alpha = ' num2str(alphas(i)) '$'],'Interpreter','latex')
    end
    hold off
end

% L, L1, L2
for idx_K=1:length(Ks)
    figure
    subplot(1,3,1)
    hold on
    plot(alphas,Ls(:, idx_K),'r*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L$','Interpreter','latex')
    subplot(1,3,2)
    hold on
    plot(alphas,L1s(:, idx_K),'b*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_1$','Interpreter','latex')
    subplot(1,3,3)
    hold on
    plot(alphas,L2s(:, idx_K),'m*-')
    xlabel('$\alpha$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

end

