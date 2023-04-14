function [mdl, L, stats] = train_kkldj(X, PiY, K, alpha, maxIters, Nrand, scaleT, verbose)
%KMEANS_ADAMAR 
arguments
    X               % Matrix of descriptors
    PiY             % Ground truth
    K               % Number of clusters
    alpha = 0.5;    % Regularization parameter
    maxIters = 10;  % Maximum number of iterations
    Nrand = 5;      % Number of random runs
    scaleT = true;
    verbose = false;
end

if verbose
    fprintf("\nPerforming the K-means algorithm for K=%d, alpha=%d\n", K, alpha);
end

%SIMULATED ANNEALING
L.L = Inf;
for nrand = 1:Nrand
    if verbose
        disp(['- annealing run #' num2str(nrand)])
    end
    
    [Lambda0, Gamma0, C0] = initial_approximation_plus_plus(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation2(X, K, PiY);
    %[Lambda0, Gamma0, C0] = initial_approximation3(X, K, PiY);
    C0=C0';
    
    [C_temp, Gamma_temp, PiX_temp, Lambda_temp, it_temp, stats_temp, L_temp] =...
    adamar_kmeans_one(C0, Gamma0, Lambda0, PiY, X, K, alpha, maxIters, scaleT, verbose);

    if L_temp.L < L.L
        C = C_temp;
        Gamma = Gamma_temp;
        PiX = PiX_temp;
        Lambda = Lambda_temp;
        it = it_temp;
        stats = stats_temp;
        L = L_temp;
    end
end

mdl = struct('C', C,...
    'Gamma', Gamma,...
    'Pi', PiX',...
    'Lambda', Lambda,...
    'it', it);

end


function [C, Gamma, PiX, Lambda, it, stats, L_out]...
    = adamar_kmeans_one(C, Gamma, Lambda, PiY, X, K, alpha, maxIters, scaleT, verbose)

if scaleT
   [T, D] = size(X);
else
    T = 1;
    D = 1;
end

% Initial objective function value
L = compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T, D);
L0 = L;
if verbose
    fprintf("it=%d  L=%.2f\n", 0, L0);
end
learningErrors = zeros(0, maxIters); % preallocation
ground_truth = PiY(1,:);
myeps = 1e-4; %TODO

for it = 1:maxIters
    
    % Compute Gamma %
    %disp([' - before Gamma: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])
    [Gamma] = akmeans_gamma_step(X, C, K, Lambda, PiY, alpha, T, D);
    %disp([' - after Gamma: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])
    
    % Update Lambda %
    %disp([' - before Lambda: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])
    Lambda = lambda_solver_jensen(Gamma, PiY);
    %disp([' - after Lambda: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])

    % Update C %
    %disp([' - before C: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])
    for k = 1:K
        ids = Gamma(k,:) == 1; % Matrix of indices of features affiliated to the k-th cluster
        if sum(ids) > 0 
            C(k,:) = mean(X(ids,:), 1);
        end
    end
    %disp([' - after C: ' num2str(compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY, T))])
    
    L_old = L;

    [L,L1,L2] = compute_fval_adamar_kmeans(C',Gamma,Lambda,X',alpha, PiY,T,D);
    [L_real,L1_real,L2_real] = compute_L2(C',Gamma,Lambda,X',alpha,PiY,T,D);

    if isnan(L)
        fprintf("\nObjective function value is NaN!\n")
        keyboard
    end
        
    Gamma_rec = compute_Gamma_kmeans(C',X'); % Reconstruction of Gamma
    %PiX = round(Lambda*Gamma_rec)'; % Prediction (round => binary matrix)
    PiX = Lambda*Gamma_rec;
    
    if size(PiY, 1) > 2 % multi-class classification
        c = size(PiY, 1); % number of classes
        PiX = round(PiX');
        R = PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1);
        r = randi([1 c],1,size(R,2));
        PiX(:, sum(PiX,1)==0 | sum(PiX,1) > 1) = bsxfun(@eq, r(:), 1:c)';
        [prediction, ~] = find(PiX);
        [ground_truth, ~] = find(round(PiY));
        if length(prediction) ~= length(ground_truth)
            keyboard
        end
        stats = statistics_multiclass(prediction, ground_truth);
        if verbose
            fprintf("it=%d  L=%.2f  L_a=%.2f  f1score=%.3f\n",...
                it, L, L_real, stats.f1score);
        end
    else % binary classification
        stats = statistics(PiX(1,:), ground_truth);
        learningErrors(it) = sum(abs(PiX(1,:) - ground_truth)) / length(ground_truth);
        if verbose
            fprintf("it=%d  L=%.2f  L_a=%.2f  FN=%d  FP=%d  f1score=%.3f  error:%.3f\n",...
                it, L, L_real, stats.fn, stats.fp, stats.f1score, learningErrors(it));
        end
    end

    
    if L_old < L
        keyboard
    end
    
    if abs(L - L_old) < myeps
        break;
    end
end

L_out.L = L;
L_out.L1 = L1;
L_out.L2 = L2;

end

