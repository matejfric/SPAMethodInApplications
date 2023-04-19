clear all
addpath(genpath(pwd));
rng(13);

%{
    find Lambda \in P^{M,K} such that 
    Pi(:,t) = Lambda*Gamma(:,t) as close as possible
    with provided Gamma \in P^{K,T}, Pi \in P^{M,T}, 
    P is set of left-stochasic matrices
%}

Ts = 100:100:1000;
K = 10; 
M = 5;

nrand = 10; % number of random runs

sigmas = 0:0.1:1; % noise parameter

% preallocation
err_L2_kld = Inf*ones(length(sigmas),nrand,length(Ts));
err_Jensen_kld = Inf*ones(length(sigmas),nrand,length(Ts));
err_KLD_kld = Inf*ones(length(sigmas),nrand,length(Ts));

err_L2_fro = Inf*ones(length(sigmas),nrand,length(Ts));
err_Jensen_fro = Inf*ones(length(sigmas),nrand,length(Ts));
err_KLD_fro = Inf*ones(length(sigmas),nrand,length(Ts));

for t = progress(1:length(Ts))
    T = Ts(t);
    for i = 1:length(sigmas)
        sigma = sigmas(i);
        for j = 1:nrand
            [Gamma_true,Pi_true,Lambda_true] = generate_synthetic_problem_Bayes(T,K,M);

            % add noise
            Gamma = projection_simplex(Gamma_true + sigma * randn(size(Gamma_true)));
            Pi = projection_simplex(Pi_true + sigma * randn(size(Pi_true)));

            % Gamma = projection_simplex(Gamma_true + sigma*(-1 + 2*rand(size(Gamma_true))));
            % Pi = projection_simplex(Pi_true + sigma * (-1 + 2*rand(size(Pi_true))));

            Lambda0 = random_Lambda(size(Pi,1),size(Gamma,1));
            
            % minimize L2 norm
            Lambda_L2 = compute_Lambda_L2(Gamma,Lambda0,Pi, 1);

            % minimize Jensen est of KLD
            Lambda_Jensen = lambda_solver_jensen(Gamma,Pi);

            % minimize KLD
            Lambda_KLD = compute_Lambda(Gamma,Lambda0,Pi, 1);

            err_L2_kld(i,j,t) = KLDiv(Lambda_L2, Lambda_true); 
            err_Jensen_kld(i,j,t) = KLDiv(Lambda_Jensen, Lambda_true); 
            err_KLD_kld(i,j,t) = KLDiv(Lambda_KLD, Lambda_true); 

            err_L2_fro(i,j,t) = norm(Lambda_L2 - Lambda_true,'fro');
            err_Jensen_fro(i,j,t) = norm(Lambda_Jensen - Lambda_true,'fro');
            err_KLD_fro(i,j,t) = norm(Lambda_KLD - Lambda_true,'fro');
        end
    end
end

[err_L2_avg, err_Jensen_avg, err_KLD_avg] = average_error(err_L2_kld, err_Jensen_kld, err_KLD_kld);
plot_synthetic_Bayes(err_L2_avg, err_Jensen_avg, err_KLD_avg, Ts, sigmas, "KLD");

[err_L2_avg, err_Jensen_avg, err_KLD_avg] = average_error(err_L2_fro, err_Jensen_fro, err_KLD_fro);
plot_synthetic_Bayes(err_L2_avg, err_Jensen_avg, err_KLD_avg, Ts, sigmas, "Frobenius Norm");


function [err_L2_avg, err_Jensen_avg, err_KLD_avg] = average_error(err_L2, err_Jensen, err_KLD)
%AVERAGE_ERROR
    % average of random runs
    err_L2_avg = squeeze(mean(err_L2,2));
    err_Jensen_avg = squeeze(mean(err_Jensen,2));
    err_KLD_avg = squeeze(mean(err_KLD,2));
end


function plot_synthetic_Bayes(err_L2_avg, err_Jensen_avg, err_KLD_avg, Ts, sigmas, label)
%PLOT_SYNTHETIC_BAYES
arguments
    err_L2_avg, err_Jensen_avg, err_KLD_avg, Ts, sigmas, label=[]
end
    [XX,YY] = meshgrid(sigmas,Ts);
    
    figure
    surf(XX,YY,err_L2_avg', 'FaceColor',[0,0.6,0], 'FaceAlpha',0.5, 'EdgeColor',[0,0.6,0])
    hold on
    surf(XX,YY,err_Jensen_avg', 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
    surf(XX,YY,err_KLD_avg', 'FaceColor','r', 'FaceAlpha',0.5, 'EdgeColor','r')
	legend('L2','Jensen','KLD')
    xlabel('$\sigma$','Interpreter','latex')
    ylabel('$T$','Interpreter','latex')
    zlabel('error')
    if ~isempty(label)
        title(label)
    end
    axis tight
    grid on
    pbaspect([1.5 1.5 2])
end


function [Gamma_true,Pi_true,Lambda_true] = generate_synthetic_problem_Bayes(T,K,M)
%GENERATE_SYNTHETIC_PROBLEM_BAYES
    Gamma_true = random_Lambda_binary(K,T);

    Lambda_true = random_Lambda(M,K);
    %Lambda_true = random_Lambda_binary(M,K);

    Piprob = Lambda_true*Gamma_true; % PiY

    edges = cumsum([zeros(1,T);Piprob],1);
    uniform = rand(1, T);

    idx = zeros(1,T);
    for i = 1:size(Piprob,1)
        idx(uniform >= edges(i,:)) = idx(uniform >= edges(i,:)) + 1;
    end

    Pi_true = zeros(size(Piprob));
    for i = 1:size(Pi_true,1)
        Pi_true(i,idx == i) = 1;
    end
end


function Lambda = random_Lambda(M,K)
%RANDOM_LAMBDA
    Lambda = rand(M,K);
    for k=1:K
        Lambda(:,k) = Lambda(:,k)/sum(Lambda(:,k));
    end
end


function Lambda = random_Lambda_binary(M,K)
%RANDOM_LAMBDA_BINARY
    Lambda = zeros(M,K);
    idx = randi(M,K,1);
    for m = 1:M
        Lambda(m,idx == m) = 1;
    end
end


function KLDivergence = KLDiv(Lambda1,Lambda2)
%MYKLD
%{
    KL(P1(x),P2(x)) =  - sum[P1(x).log(P2(x)/P1(x))]
%}
    KLDivergence = 0;    
    for k = 1:size(Lambda1,2)
        pi1 = Lambda1(:,k);
        pi2 = Lambda2(:,k);
        KLDivergence = KLDivergence - ...
            dot(pi1(pi1 ~= 0), mylog(pi2(pi1 ~= 0) ./ pi1(pi1 ~= 0))); 
    end
end

