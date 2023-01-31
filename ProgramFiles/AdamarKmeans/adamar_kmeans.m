function [Lambda, C, Gamma, stats, L, PiX] = adamar_kmeans(X, K, alpha, maxIters)
%KMEANS_ADAMAR 
%   K........number of clusters
switch(nargin)
    case 2 
        alpha = 0.5;
        maxIters = 10;
    case 3
        maxIters = 10;
end

fprintf("\nPerforming the K-means algorithm for K=%d, alpha=%d\n", K, alpha);

PiY = [X(:,end)'; 1-X(:,end)']; % [P(x is corroded); P(x is not corroded)]
X = X(:, 1:end-1);

%SIMULATED ANNEALING
L.L = Inf;
Nrand = 5; % Number of random runs

for nrand = 1:Nrand
    disp(['- annealing run #' num2str(nrand)])
    
    [Lambda0, Gamma0, C0] = initial_approximation_plus_plus(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation2(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation3(X, K, PiY);
    C0=C0';
    
    [Lambda_temp, C_temp, Gamma_temp, PiX_temp, stats_temp, L_temp] =...
    adamar_kmeans_one(C0, Gamma0, Lambda0, PiY, X, K, alpha, maxIters);

    if L_temp.L < L.L
        C = C_temp;
        Gamma = Gamma_temp;
        PiX = PiX_temp;
        Lambda = Lambda_temp;
        stats = stats_temp;
        L = L_temp;
    end
end

end


function [Lambda, C, Gamma, PiX, stats, L_out]...
    = adamar_kmeans_one(C, Gamma, Lambda, PiY, X, K, alpha, maxIters)

% Initial objective function value
T = size(X,1); % Number of features
L = compute_L2(C',Gamma,Lambda,X',alpha, PiY, T);
L0 = L;
fprintf("it=%d  L=%.2f\n", 0, L0);
learningErrors = zeros(0, maxIters); % preallocation
ground_truth = PiY(1,:);
myeps = 1e-3; %TODO

for i = 1:maxIters
    
    %disp([' - before Gamma: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    %Compute Gamma
    [Gamma] = akmeans_gamma_step(X, C, K, Lambda, PiY, alpha);
    %disp([' - after Gamma: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    
    % Update Lambda
    Lambda = lambda_solver_jensen(Gamma, PiY);

    %disp([' - before C: ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY))])
    % Update C
    for k = 1:K
        ids = Gamma(k,:) == 1; % Matrix of indices of features affiliated to the k-th cluster
        if sum(ids) > 0 
            C(k,:) = mean(X(ids,:));
        end
    end

    %disp([' - L = ' num2str(compute_L2(C',Gamma,Lambda,X',alpha, PiY)) ', L_real: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY))])
    % Is the objective function decreasing?
    L_old = L;
    [L,L1,L2] = compute_L2(C',Gamma,Lambda,X',alpha, PiY, T);
    if isnan(L) % Only one state is present in Lambda!
        fprintf("\nObjective function value is NaN!\n")
        
        keyboard
        
        stats = statistics(zeros(numel(ground_truth),1), ground_truth);
        pause(0.1)
        break;
    end
    
    if L_old < L
        break;
    end
    
    if abs(L - L_old) < myeps
        break;
    end
    
    [L_real] = compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T);
    
%    if isnan(L_real)
%        keyboard
%    end
        
    % Computation of learning error
    Gamma_rec = compute_Gamma_kmeans(C',X'); % Reconstruction of Gamma
    %PiX = round(Lambda*Gamma_rec)'; % Prediction (round => binary matrix)
    PiX = (Lambda*Gamma_rec)';
    stats = statistics(PiX(:,1), ground_truth);
    learningErrors(i) = sum(abs(PiX(:,1) - ground_truth')) / length(ground_truth);
    fprintf("it=%d  L=%.2f  L_a=%.2f  FN=%d  FP=%d  f1score=%.3f  error:%.3f\n",...
        i, L_real, L, stats.fn, stats.fp, stats.f1score, learningErrors(i));
end

L_out.L = L_real;
L_out.L1 = L1;
L_out.L2 = L2;

end

