function [] = adamar_helper(X, K, descriptors)
%ADAMAR_HELPER Summary of this function goes here
%   K........number of clusters

% Remember scaling for testing dataset
a = min(X(:, 1));
b = max(X(:, 1));
%X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Initial approximation of C and Lambda
[idx, C] = kmeans(X(:,1:end-1), K, 'MaxIter',1000);
Gamma = zeros(K,length(idx));
for k = 1:K
    Gamma(k,idx==k) = 1;
end
PiY = [X(:,end)'; 1-X(:,end)'];
Lambda = lambda_solver_jensen(Gamma, PiY);

% PiY (2 x T), resp. (K_Y, T)
% Gamma=PiX (K, T), resp. (K_X, T)
% Lambda (2 x K), resp. (K_Y, K_X)
% C (K, 4), resp. (K, features)
maxIters = 2;
[C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train] = ...
    adamar_fmincon(normalize(X(:,1:end-1)'), K, 1e-4, C', Gamma, Lambda, PiY, X(:,end), maxIters);

disp(Lambda);

for i = 1:maxIters
    lprecision(i) = stats_train(i).precision;
    lrecall(i) = stats_train(i).recall;
    lf1score(i) = stats_train(i).f1score;
    laccuracy(i) = stats_train(i).accuracy;
end
    
images = [ 172, 177, 179, 203];
for i= 1:numel(images)
    [stats_test] = adamar_predict(Lambda, C, K, [], [], images(i), descriptors);
    
    tprecision(i) = stats_test.precision;
    trecall(i) = stats_test.recall;
    tf1score(i) = stats_test.f1score;
    taccuracy(i) = stats_test.accuracy;
end

% PLOTS

% Training
range = 1:maxIters;

figure
subplot(1,2,1)
set(gca,'DefaultLineLineWidth',2)
plot(range,lprecision) 
hold on 
plot(range,lrecall)
hold on 
plot(range,lf1score)
hold on 
plot(range,laccuracy)
xlabel('Iteration')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Training Phase')
hold off

% Testing
range = 1:numel(images);

subplot(1,2,2)
set(gca,'DefaultLineLineWidth',2)
plot(range,tprecision) 
hold on 
plot(range,trecall)
hold on 
plot(range,tf1score)
hold on 
plot(range,taccuracy)
xlabel('Image')
ylabel('Score')
legend('Precision','Recall', 'F1-score', 'Accuracy')
title('Testing Phase')
sgtitle('Adamar fmincon()')
hold off

end

