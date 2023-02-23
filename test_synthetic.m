clear all
close all

addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
addpath('ProgramFiles/SPG') 

T = 4;
[X_true,Y_true,C_true,Gamma_true,Lambda_true] = generate_synthetic_problem(T);

sigma = 10.1; % noise parameter
X = X_true + sigma * randn(size(X_true));

Y = Y_true;
nwrong = floor(0.0*T); % wrong labels
idx = randperm(T);
for i=1:nwrong
  Y(1:2,idx(i)) = Y(2:-1:1,idx(i));
end

maxIters = 5e2;
Ks = 4;%size(C_true,2);

myepss = 10.^[-1:0.1:6];
%alphas = 0.9:0.01:1;
%alphas = 0.05:0.05:0.95;
Ls  = zeros(numel(myepss),length(Ks));
L1s = zeros(numel(myepss),length(Ks));
L2s = zeros(numel(myepss),length(Ks));

for idx_myeps=1:length(myepss)
    myeps = myepss(idx_myeps);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
    
            [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X', Y, K, myeps, maxIters,1e1);
%            [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L_out] = adamar_fmincon(X', Y, K, myeps, maxIters);
            
%             [prediction, ~] = find(round(Lambda * Gamma));
%             [ground_truth, ~] = find(Y_true);
%             fprintf("f1score: %.2f", statistics_multiclass(prediction, ground_truth).f1score);

            Ls(idx_myeps,idx_K)  = L_out.L;
            L1s(idx_myeps,idx_K) = L_out.L1;
            L2s(idx_myeps,idx_K) = L_out.L2;
        end
end

% L-curve
for idx_K=1:length(Ks)
    figure
    hold on
    title(sprintf('K=%d', K))
    plot(L1s(:,idx_K), L2s(:,idx_K),'r-o');
    %text(L1s(1),L2s(1),['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    %text(L1s(end),L2s(end),['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    for i = 1:numel(L1s(:,idx_K))
        text(L1s(i),L2s(i),['$\epsilon = ' num2str(myepss(i)) '$'],'Interpreter','latex')
    end
    xlabel('$L_1$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end

for idx_K=1:length(Ks)
    figure
    subplot(1,3,1)
    hold on
    plot(myepss,Ls(:, idx_K),'r*-')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L$','Interpreter','latex')
    subplot(1,3,2)
    hold on
    plot(myepss,L1s(:, idx_K),'b*-')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L_1$','Interpreter','latex')
    subplot(1,3,3)
    hold on
    plot(myepss,L2s(:, idx_K),'m*-')
    xlabel('$\epsilon$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end
