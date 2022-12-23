function [error, count, stats] = adamar_predict(Lambda, C, K, a, b, image_number)
%ADAMAR_PREDICT Make a prediction based on ADAMAR model
%   Lambda...transion matrix
%   C........centroids
%   K........number of clusters
%   a........min of X(:,1)
%   b........max of X(:,1)

if(isempty(image_number))
    %image_number = 137; % 137...hřebík
    image_number = 68;
end

ca_predict = cell(1,2);
ca_predict{1,1} = imread(sprintf('Dataset/Original/%d.jpg', image_number));
ca_predict{1,2} = imread(sprintf('Dataset/Annotations/%d.png', image_number));
Y = [roughness_analysis(ca_predict),color_analysis(ca_predict), get_ground_truth(ca_predict)];
%Y(:, 1) = (Y(:, 1) - a) / (a - b); % Scale to [-1,1] (MinMaxScaler)

% Kmeans
dist_from_C = zeros(K, size(Y,1));
for k=1:K
    dist_from_C(k,:) = sum((Y(:,1:end-1)' - C(:,k)).^2,1);
end
[~,idxY] = min(dist_from_C);
GammaY = zeros(K,length(idxY));
for k = 1:K
   GammaY(k,idxY==k) = 1; 
end

piY = round(Lambda*GammaY)'; % round => binary matrix
errY = sum(abs(piY(:,1) - Y(:,end))) / size(Y,1);
%disp(['Testing error = ' num2str(errY)]);
count = [sum(piY(:,1)), sum(Y(:,end))];
error = errY;

stats = statistics(piY(:,1), Y(:,end));

end

