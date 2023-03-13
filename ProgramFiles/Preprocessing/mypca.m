function [XTrain95, ca_XYTest, explained] = mypca(XTrain, ca_XYTest)
%PRINCIPAL_COMPONENT_ANALYSIS
%https://www.mathworks.com/help/stats/pca.html#:~:text=requires%20MATLAB%C2%AE%20Coder%E2%84%A2.-,Apply%20PCA,-to%20New%20Data
%https://www.mathworks.com/matlabcentral/answers/270329-how-to-select-the-components-that-show-the-most-variance-in-pca#comment_1302615
arguments
    XTrain (:,:) double
    ca_XYTest = []
end

[coeff,scoreTrain,~,~,explained,mu] = pca(XTrain);

%Find the number of components required to explain at least 95% variability.
idx = find(cumsum(explained)>95,1);
XTrain95 = scoreTrain(:,1:idx);

% Visualization of the first 3 principal components
% figure
% scatter3(scoreTrain(:,1),scoreTrain(:,2),scoreTrain(:,3))
% axis equal
% xlabel('1st Principal Component')
% ylabel('2nd Principal Component')
% zlabel('3rd Principal Component')  

n = numel(ca_XYTest);
for i = 1:n
    XYTest = ca_XYTest{i}.X;
    
    XTest = XYTest(:,1:end-1);
    YTest = XYTest(:,end);
 
    XTest95 = (XTest-mu)*coeff(:,1:idx);
    
    ca_XYTest{i}.X = [XTest95, YTest];
end

end

