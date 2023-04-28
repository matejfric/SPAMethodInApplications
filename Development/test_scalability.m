clear all
addpath(genpath(pwd));
rng(42);

classifiers{1} = KLambda();
classifiers{2} = KKLDJ();
classifiers{3} = KKLD();
% classifiers{4} = SPAKLD();
%clf_names = pad(["KLambda", "KKLDJ", "KKLD", "SPAKLD", "KNN", "SVM", "D-Tree"],'both');
clf_names = ["K-means & Lambda", "K-means+KLD+Jensen", "K-means+KLD", "SVM", "KNN"];


[X_true,Pi_true,C_true,Gamma_true,Lambda_true] = generate_binary_synthetic_problem(10);
%X_true = scaling(X_true, [], 'minmax');

n_rand = 5;
alpha = 0.5;
[M,K] = size(Lambda_true);
Ts = [10.^(2:1:5), 2*10.^5];

% Set hyperparameters
for clf=1:length(classifiers)
    classifiers{clf}.alpha = alpha;
    classifiers{clf}.K = K; 
end

% Test
tmp_time = Inf*ones(n_rand,1);
times = Inf*ones(length(classifiers),length(Ts));

for t = progress(1:length(Ts))
    
    T = Ts(t);
    [X_train,Pi_train] = generate_binary_synthetic_problem(T);
    y_train = myonehotdecode(Pi_train,1:M);
    
    for clf=1:length(classifiers)
        for n = 1:n_rand
            tic;
            classifiers{clf}.fit(X_train',y_train')
            tmp_time(n) = toc;
        end     
        
        times(clf, t) = mean(tmp_time);
    end
    
    for clf=length(classifiers)+1:length(classifiers)+2
        for n = 1:n_rand
            tic;
            mdlSVM = fitcecoc(X_train',y_train');
            tmp_time(n) = toc;
        end     
        
        times(clf, t) = mean(tmp_time);
    end
    
    for clf=length(classifiers)+2:length(classifiers)+3
        for n = 1:n_rand
            tic;
            mdlKNN = fitcknn(X_train',y_train');
            tmp_time(n) = toc;
        end     
        
        times(clf, t) = mean(tmp_time);
    end
end

% Plotting
figure;
plot(Ts, times', '-o', 'linewidth', 2);
xlabel('Dimension $T$', 'interpreter', 'latex');
ylabel('Execution Time [$s$]', 'interpreter', 'latex');
set(gca, 'XScale', 'log')
set(gca, 'fontsize', 14)
legend(clf_names, 'location', 'northwest');
grid on
grid minor

