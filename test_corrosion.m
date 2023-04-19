%ADAMAR FMINCON()
close all
clear all
addpath(genpath(pwd));
rng(42);

DATASET = 'DatasetSelection'; % 'Dataset', 'Dataset2', 'Dataset256', 'DatasetSelection'
VISUALIZE = false;

[X, ca_Y] = get_train_test_data(DATASET);

%[X, ca_Y] = correlation_analysis(X, ca_Y); % Removal of strongly correlated columns

PiY = [X(:,end), 1-X(:,end)]';
X = X(:,1:end-1);

[X, ca_Y] = scaling(X, ca_Y, 'minmax');
%[X, ca_Y] = scaling(X, ca_Y, 'zscore', 'robust');

%Standardizing is usually done when the variables
%on which the PCA is performed are not measured on the same scale.
%Note that standardizing implies assigning equal importance to all variables.
[X, ca_Y] = mypca(X, ca_Y);

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n ", sum(PiY(1,:)), sum(PiY(2,:)));

epsilons = 10.^(1:1:6);
Ks = 32; % Number of clusters
maxIters = 25;
nrand = 3;
% [L_out1, L_out2, L_out3, L_out4, stats_train1, stats_train2, stats_train3, stats_train4, ...
%     stats_test1, stats_test2, stats_test3, stats_test4] = preallocation(epsilons, Ks);

for a = 1:numel(epsilons)
    epsilon = epsilons(a);
    
    for k = 1 : length(Ks)
        K = Ks(k);
        
        %[C1, Gamma1, PiX1, Lambda1, it1, stats_train1(a,k), L_out1(a,k)] = adamar_fmincon(X, PiY, K, epsilon, maxIters, nrand);
        [C2, Gamma2, PiX2, Lambda2, it2, stats_train2(a,k), L_out2(a,k)] = kmeans_lambda(X, PiY, K, epsilon);
        [C3, Gamma3, PiX3, Lambda3, it3, stats_train3(a,k), L_out3(a,k)] = adamar_kmeans(X, PiY, K, epsilon, maxIters, nrand);
        %[C4, Gamma4, PiX4, Lambda4, it4, stats_train4(a,k), L_out4(a,k)] = adamar_spa(X, PiY, K, epsilon, maxIters, nrand);

        %[stats_test1(a,k)] = adamar_predict(Lambda1, C1, K, epsilon, ca_Y, DATASET, VISUALIZE);
        [stats_test2(a,k)] = adamar_predict(Lambda2, C2', K, epsilon, ca_Y, DATASET, VISUALIZE);
        [stats_test3(a,k)] = adamar_predict(Lambda3, C3', K, epsilon, ca_Y, DATASET, VISUALIZE);
        %[stats_test4(a,k)] = adamar_predict(Lambda4, C4, K, epsilon, ca_Y, DATASET, VISUALIZE);
    end
end

%regularization_plot(sprintf('Adamar SPG, k=%d', Ks), epsilons, lprecision, lrecall, lf1score, laccuracy,tprecision, trecall, tf1score, taccuracy)

%plot_L_curves(Ls, L1s, L2s, Ks, epsilons);

% L-curve
for idx_K=1:length(Ks)
    figure
    grid on;
    hold on
    title(sprintf('K=%d', K))
    plot( [L_out1(:,idx_K).L1], [L_out1(:,idx_K).L2],'r-o');
    plot( [L_out2(:,idx_K).L1], [L_out2(:,idx_K).L2],'b-o');
    plot( [L_out3(:,idx_K).L1], [L_out3(:,idx_K).L2],'m-o');
    plot( [L_out4(:,idx_K).L1], [L_out4(:,idx_K).L2],'g-o');

    xlabel('$L_1$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    set(gca, 'XScale', 'log')
    legend('fmincon','bayes', 'jensen', 'spa')
    hold off
end

for idx_K=1:length(Ks)
    figure
    subplot(1,3,1)
    grid on;
    hold on
    plot(epsilons,[L_out1(:,idx_K).L],'r*-')
    plot(epsilons,[L_out2(:,idx_K).L],'b*-')
    plot(epsilons,[L_out3(:,idx_K).L],'m*-')
    plot(epsilons,[L_out4(:,idx_K).L],'g*-')
    set(gca, 'XScale', 'log')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L$','Interpreter','latex')
    legend('fmincon','bayes','jensen', 'spa')
    
    subplot(1,3,2)
    grid on;
    hold on
    plot(epsilons,[L_out1(:,idx_K).L1],'r*-')
    plot(epsilons,[L_out2(:,idx_K).L1],'b*-')
    plot(epsilons,[L_out3(:,idx_K).L1],'m*-')
    plot(epsilons,[L_out4(:,idx_K).L1],'g*-')
    set(gca, 'XScale', 'log')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L_1$','Interpreter','latex')
    
    subplot(1,3,3)
    grid on;
    hold on
    plot(epsilons,[L_out1(:,idx_K).L2],'r*-')
    plot(epsilons,[L_out2(:,idx_K).L2],'b*-')
    plot(epsilons,[L_out3(:,idx_K).L2],'m*-')
    plot(epsilons,[L_out4(:,idx_K).L2],'g*-')
    set(gca, 'XScale', 'log')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

% F1-score
for idx_K=1:length(Ks)
    figure
    subplot(1,2,1)
    hold on
    title(sprintf('K=%d', K))
    plot( epsilons, [stats_train1(:,idx_K).f1score],'r-o');
    plot( epsilons, [stats_train2(:,idx_K).f1score],'b-o');
    plot( epsilons, [stats_train3(:,idx_K).f1score],'m-o');
    plot( epsilons, [stats_train4(:,idx_K).f1score],'g-o');
    set(gca, 'XScale', 'log')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$f_1-score$','Interpreter','latex')
    legend('fmincon','bayes', 'jensen', 'spa')
    grid on;
    grid minor;
    
    subplot(1,2,2)
    hold on
    plot( epsilons, [stats_test1(:,idx_K).f1score],'r-o');
    plot( epsilons, [stats_test2(:,idx_K).f1score],'b-o');
    plot( epsilons, [stats_test3(:,idx_K).f1score],'m-o');
    plot( epsilons, [stats_test4(:,idx_K).f1score],'g-o');
    set(gca, 'XScale', 'log')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$f_1-score$','Interpreter','latex')
    grid on;
    grid minor;
    hold off
end


function [L_out1, L_out2, L_out3, L_out4, ...
    stats_train1, stats_train2, stats_train3, stats_train4, ...
    stats_test1, stats_test2, stats_test3, stats_test4] = preallocation(epsilons, Ks)

L_out1 = arrayfun(@(L,L1,L2)struct('L',L,'L1',L1,'L2',L2),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
L_out2 = arrayfun(@(L,L1,L2)struct('L',L,'L1',L1,'L2',L2),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
L_out3 = arrayfun(@(L,L1,L2)struct('L',L,'L1',L1,'L2',L2),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
L_out4 = arrayfun(@(L,L1,L2)struct('L',L,'L1',L1,'L2',L2),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));

stats_train1 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_train2 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_train3 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_train4 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));

stats_test1 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_test2 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_test3 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));
stats_test4 = arrayfun(@(fp,fn,precision,recall,f1score,accuracy)struct('fp',fp,'fn',fn,'precision',precision, 'recall', recall, 'f1score', f1score, 'accuracy', accuracy),zeros(numel(epsilons),length(Ks)),zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)), zeros(numel(epsilons),length(Ks)));

end

