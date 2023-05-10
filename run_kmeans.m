%--------------------------------------------------------------------------
% K-means examples
%--------------------------------------------------------------------------

clear all;
rng(42);
addpath(genpath(pwd));

% Read data (2D points)
X = readtable('kmeans_data5.csv'); K = 5;
%X = readtable('kmeans_data3.csv'); K = 3;
X = table2array(X(:,1:2));

% Plot sum of squares error for K = {1,2,...,10}
sse_elbow_method(X);

% Plot K-means++ clustering 
[C, Gamma, sse, it] = mykmeans(X', K); C=C';
plot_kmeans(X, Gamma, C, K);

%--------------------------------------------------------------------------
% Lloyd's K-means animation
%--------------------------------------------------------------------------

if true
    frames = animate_kmeans(X', K); % black background
    %frames = animate_kmeans_white(X', K); % white background
    filename = 'myanimation.gif';
    %save_animation_gif(frames, filename, 1)
end

