%--------------------------------------------------------------------------
% SPA Visualization
%--------------------------------------------------------------------------
clear all;
addpath(genpath(pwd));

X = readtable('kmeans_data3.csv'); K = 3;
X = table2array(X(:,1:2));

tic
[S, Gamma] = myspa(X',K);
toc

myspa_plot(X', Gamma, S, K)
myspa_plot2(X', Gamma, S, K)
