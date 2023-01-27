function [stats_train, stats_test] =...
    kmeans_lambda(X, K, testing_images, descriptors, testing_matrices)
%KMEANS_KL Summary of this function goes here
arguments
    X {mustBeNumeric}
    K (1,1) {mustBeNumeric}
    testing_images {mustBeNumeric} = 68
    descriptors = []
    testing_matrices = []
end

% Remember scaling for testing dataset
% a = min(X(:, 1));
% b = max(X(:, 1));
%X(:, 1) = (X(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% K-means
[idx, C] = kmeans(X(:,1:end-1), K, 'MaxIter',1000);
Gamma = zeros(K,length(idx));
for k = 1:K
   Gamma(k,idx==k) = 1; 
end

%Lambda
true_labels = X(:,end);
PiY = [ true_labels'; 1-true_labels'];
Lambda = lambda_solver_jensen(Gamma,PiY);

% Computation of learning error
PiX = round(Lambda*Gamma)'; % round => binary matrix
stats_train = statistics(PiX(:,1), X(:,end));
% errX = sum(abs(PiX(:,1) - X(:,end)))/size(X,1);
% disp(['K-means+KL+Jensen learning error = ' num2str(errX)]);

if isempty(testing_matrices)
    %Y = roughness_analysis(ca_predict);
    ca_pred = load_images(testing_images);
    Y = get_descriptors(ca_pred, descriptors);
    %Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

    % K-means
    dist_from_C = zeros(K, size(Y,1));
    for k=1:K
        dist_from_C(k,:) = sum((Y(:,1:end-1)' - C(k,:)').^2,1);
    end
    [~,idxY] = min(dist_from_C);
    GammaY = zeros(K,length(idxY));
    for k = 1:K
       GammaY(k,idxY==k) = 1; 
    end

    % Predicition
    PiY = round(Lambda*GammaY)'; % round => binary matrix
    stats_test = statistics(PiY(:,1), Y(:,end));
    % errY = sum(abs(PiY(:,1) - Y(:,end)))/size(Y,1);
    % disp(['K-means+KL+Jensen testing error  = ' num2str(errY)]);

    visualize(ca_pred(1,1), ca_pred(1,2), Y(:,end), PiY(:,1), ["K-means Lambda K=", num2str(K)]);
else
    [stats_test] = adamar_predict_mat(Lambda, C', K, [], [], testing_matrices, 'Dataset2');
end

end

