function [frames, C, Gamma, sse, it] = animate_kmeans_white(X, K, maxIters, eps)
%--------------------------------------------------------------------------
% Animate K-means - with white background
%--------------------------------------------------------------------------
%   X ....... D x T matrix
%   K ....... number of clusters
%   eps...... tolerance
%   iters ... number of iterations
%--------------------------------------------------------------------------
% This function write the animation as sequence of PDF files into folder
% 'frames/'.
%--------------------------------------------------------------------------
arguments
    X (:,:) {double}
    K {mustBeNumeric} = 5
    maxIters {mustBeNumeric} = 1e3
    eps {mustBeNumeric} = 1e-6
end

sse = zeros(1,maxIters); sse(1) = Inf; % Sum of squares error
T = size(X,2); % Number of data points in the dataset
%[C] = get_kmeans_pp_centroids(X, K); % K-means++ initialization
[C] = init_random(X', K); C = C';

% Instantiate a figure
fig = figure;
xlabel('$x$','Interpreter','latex','FontSize', 14);
ylabel('$y$','Interpreter','latex','FontSize', 14);
grid on
grid minor

frames(1) = plot_initialization(X, C);

% Save the figure as PDF
% Inspired by: Dr. Erol Kalkan, P.E. (2023).
% Crop and save MatLAB figure as PDF (savePDF)
% (https://www.mathworks.com/matlabcentral/fileexchange/70349-crop-and-save-matlab-figure-as-pdf-savepdf),
% MATLAB Central File Exchange. Retrieved May 10, 2023.
outputDir = 'frames/';
if ~isfolder(outputDir)
    mkdir(outputDir);
end
filenamePrefix = 'frame';
fileExtension = '.pdf';
outputFilename = [filenamePrefix, num2str(1), fileExtension];
outputFilePath = fullfile(outputDir, outputFilename);
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(fig,'-dpdf','-painters','-r600','-bestfit',outputFilePath);

for it = 2:maxIters+1
        Gamma = zeros(K, T);
        for t = 1:T
            [sse_t,id] = min( sum( (C - X(:,t) ).^2 , 1 ));
            sse(it) = sse(it) + sse_t;
            Gamma(id,t) = 1;
        end
        for k = 1:K
            ids = Gamma(k,:) == 1;
            C(:,k) = mean(X(:,ids),2);
        end
        
        frames(it) = plot_kmeans(X', Gamma, C', K, it-1);
        
        % Save the figure as PDF
        outputFilename = [filenamePrefix, num2str(it), fileExtension];
        outputFilePath = fullfile(outputDir, outputFilename);
        fig = gcf;
        fig.PaperPositionMode = 'auto';
        fig_pos = fig.PaperPosition;
        fig.PaperSize = [fig_pos(3) fig_pos(4)];
        print(fig,'-dpdf','-painters','-r600','-bestfit',outputFilePath);
        
        if norm(sse(it-1) - sse(it)) < eps
            sse = nonzeros(sse(2:end));
            it = it-1;
            break
        end
end
    
end

function [frame, myframe] = plot_kmeans(X, Gamma, centroids, K, iter)
%PLOT_KMEANS

if K==5
    colors = [[0 0.4470 0.7410];
              [0.4660 0.6740 0.1880];
              [0.3010 0.7450 0.9330]; 
              [0.9290 0.6940 0.1250];
              [1 0.6 0.8]];
else
    colors = hsv(K);
    %colors = jet(K);
end

clusters = string.empty;

% Draw clusters
for i = 1:K
        ids = Gamma(i,:)==1;
        scatter(X(ids,1), X(ids,2), 30,...
            'MarkerFaceColor',colors(i,:), 'Marker', "diamond",...
            'MarkerEdgeColor',colors(i,:), 'LineWidth',1,...
            'MarkerFaceAlpha', 0.4)
        hold on
        clusters(i)= string(i);
end

% Plot the Voronoi diagram
h = voronoi(centroids(:,1), centroids(:,2));
set(h, 'LineWidth', 2,'Color', [0.7,0.7,0.7], 'HandleVisibility', 'off');

% Draw centroids
scatter(centroids(:,1),centroids(:,2),40 ,...
    'MarkerEdgeColor','r','MarkerFaceColor','r',...
    'MarkerFaceAlpha', 0.9)

hold off

% Legend and title
lgd = legend(clusters,'Location', 'eastoutside', 'Orientation', 'vertical',...
    'NumColumns', 1);
title(lgd,'Cluster ID')
title(sprintf('$K$-means iteration: %d', iter),...
    'Interpreter','Latex', 'FontSize',13);

myframe = getframe(gcf);
im = frame2im(myframe);
[cdata,colormap] = rgb2ind(im,256);
frame.cdata = cdata;
frame.colormap = colormap;

end

function [frame, myframe] = plot_initialization(X, C)
%PLOT_INITIALIZATION Plot initial centroids for the k-means algorithm
% X ... D, T
% C ... D, K

blue = [0 0.4470 0.7410];

% Draw points
scatter(X(1,:), X(2,:), 30,'Marker', 'diamond', ...
    'MarkerEdgeColor',blue,'MarkerFaceColor',blue,...
    'MarkerFaceAlpha', 0.4, 'LineWidth', 1)
hold on

% Draw centroids
scatter(C(1,:),C(2,:),40 ,'MarkerEdgeColor','r','MarkerFaceColor','r',...
    'MarkerFaceAlpha', 0.9)

hold off

% Legend and title
lgd = legend("1",'Location', 'eastoutside', 'Orientation', 'vertical',...
    'NumColumns', 1);
title(lgd,'Cluster ID')
title('$K$-means random centroid initialization','Interpreter','Latex',...
    'FontSize',13);


myframe = getframe(gcf);
im = frame2im(myframe);
[cdata,colormap] = rgb2ind(im,256);
frame.cdata = cdata;
frame.colormap = colormap;

end


