function [] = plot_f1score(stats_train, stats_test, epsilons, K)
%PLOT_STATS Summary of this function goes here
%   Detailed explanation goes here

% F1-score
figure
subplot(1,2,1)
hold on
title(sprintf('K=%d', K))
plot( epsilons, [stats_train.f1score],'b-*');
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('$f_1-score$','Interpreter','latex')
set(gca, 'XScale', 'log')
grid on;
grid minor;

subplot(1,2,2)
hold on
plot( epsilons, [stats_test.f1score],'b-*');
xlabel('$\varepsilon$','Interpreter','latex')
ylabel('$f_1-score$','Interpreter','latex')
set(gca, 'XScale', 'log')
grid on;
grid minor;
hold off

end

