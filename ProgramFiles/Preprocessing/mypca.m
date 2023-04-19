function [XTrain95, ca_XYTest, PCA] = mypca(XTrain, ca_XYTest, ground_truth)
%PRINCIPAL_COMPONENT_ANALYSIS
%https://www.mathworks.com/help/stats/pca.html#:~:text=requires%20MATLAB%C2%AE%20Coder%E2%84%A2.-,Apply%20PCA,-to%20New%20Data
%https://www.mathworks.com/matlabcentral/answers/270329-how-to-select-the-components-that-show-the-most-variance-in-pca#comment_1302615
arguments
    XTrain (:,:) double
    ca_XYTest = []
    ground_truth = []
end

[coeff,scoreTrain,~,~,explained,mu] = pca(XTrain);

% Plot
%plot_pca(coeff, scoreTrain, explained, ground_truth);

% Find the number of components required to explain at least 95% variability.
idx = find(cumsum(explained)>95,1); % hyperparameter
XTrain95 = scoreTrain(:,1:idx);

PCA = struct('explained', explained, 'coeff', coeff, 'mu', mu, 'idx', idx);

n = numel(ca_XYTest);
for i = 1:n
    XYTest = ca_XYTest{i}.X;
    
    XTest = XYTest(:,1:end-1);
    YTest = XYTest(:,end);
 
    XTest95 = (XTest-mu)*coeff(:,1:idx);
    
    ca_XYTest{i}.X = [XTest95, YTest];
end

end

function plot_pca(coeff, scoreTrain, explained, ground_truth)

if isempty(ground_truth)
    fprintf('Missing ground truth in "plot_pca.m". Returning without visualization.\n');
    return
end

figure()
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');

% BiPlot
subplot(1,3,1)
x = 1:size(coeff, 1);
labels = split(sprintf([repmat('v_%d ',1,numel(x)) '%d'], x), ' ');
%figure()
h = biplot(coeff(:,1:2),'scores',scoreTrain(:,1:2),'varlabels',labels(1:end-1));
% Identify each handle
hID = get(h, 'tag'); 
% Isolate handles to scatter points
hPt = h(strcmp(hID,'obsmarker')); 
% Identify cluster groups
clusters = ground_truth;
grp = findgroups(clusters);    %r2015b or later - leave comment if you need an alternative
grp(isnan(grp)) = max(grp(~isnan(grp)))+1; 
grpID = 1:max(grp); 
% assign colors and legend display name
clrMap = lines(length(unique(grp)));   % using 'lines' colormap
for i = 1:max(grp)
    set(hPt(grp==i), 'Color', clrMap(i,:), 'DisplayName', sprintf('Cluster %d', grpID(i)))
end
% add legend to identify cluster
[~, unqIdx] = unique(grp);
legend(hPt(unqIdx))

% Visualization of the first 3 principal components
if false
    subplot(1,3,2)
    %figure
    scatter3(scoreTrain(:,1),scoreTrain(:,2),scoreTrain(:,3),'+')
    axis equal
    xlabel('1st Principal Component')
    ylabel('2nd Principal Component')
    zlabel('3rd Principal Component')  
end

% Scree Plot
subplot(1,3,3)
%figure
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')

end

