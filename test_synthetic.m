clear all

addpath('Dataset0')
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon') % adamar_predict()
addpath('ProgramFiles/AdamarKmeans') % adamar_kmeans
addpath('ProgramFiles/SPG') 

T = 100;
[X,Y,C_true,Gamma_true,Lambda_true] = generate_problem(T);

sigma = 0; % noise parameter
X = X + sigma*randn(size(X));

nwrong = floor(0.2*T);
idx = randperm(T);
for i=1:nwrong
  Y(1:2,idx(i)) = Y(2:-1:1,idx(i));
end

maxIters = 30;
Ks = size(C_true,2);

alphas = 0:0.05:1;
L1s = zeros(numel(alphas),length(Ks));
L2s = zeros(numel(alphas),length(Ks));

for idx_alpha=1:length(alphas)
    alpha = alphas(idx_alpha);
    
        for idx_K=1:length(Ks)
            K = Ks(idx_K);
    
            [Lambda, C, Gamma, stats_train, L_out, PiX] = adamar_kmeans(X', Y, K, alpha, maxIters);
%            [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats, L_out] = adamar_fmincon(X', Y, K, alpha, maxIters);

            L1s(idx_alpha,idx_K) = L_out.L1;
            L2s(idx_alpha,idx_K) = L_out.L2;
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
        text(L1s(i),L2s(i),['$\alpha = ' num2str(alphas(i)) '$'],'Interpreter','latex')
    end
    xlabel('$L_1$','Interpreter','latex')
    ylabel('$L_2$','Interpreter','latex')
    hold off
end
