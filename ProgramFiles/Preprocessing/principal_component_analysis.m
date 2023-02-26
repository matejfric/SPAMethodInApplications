function [X, ca_Y] = principal_component_analysis(X, ca_Y, num_of_dimensions)
%PRINCIPAL_COMPONENT_ANALYSIS
%Inspired by: https://www.maskaravivek.com/post/principal-component-analysis-in-matlab/
arguments
    X (:,:) double
    ca_Y
    num_of_dimensions = 15 % No of dimensions to keep
end

ground_truth = X(:,end);

% Normalize the feature matrix
X = normalize(X(:,1:end-1)); 

% De-mean
X = bsxfun(@minus,X,mean(X));

% Perform the PCA
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(X);

reduced_dimension = COEFF(:,1:num_of_dimensions);
X_reduced = X * reduced_dimension;
X_reduced(:,end+1) = ground_truth;

[X] = X_reduced;     

n = numel(ca_Y);
for i = 1:n
    Y = ca_Y{i}.X;
    ground_truth = Y(:,end);
    % Normalize the feature matrix
    Y = normalize(Y(:,1:end-1));
    % De-mean
    Y = bsxfun(@minus, Y, mean(Y));
    % Perform the PCA
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED] = pca(Y);
    reduced_dimension = COEFF(:,1:num_of_dimensions);
    Y_reduced = Y * reduced_dimension;
    Y_reduced(:,end+1) = ground_truth;
    ca_Y{i}.X = Y_reduced;
end

end

