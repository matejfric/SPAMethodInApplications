%ADAMAR FMINCON()
close all
clear all
addpath('ProgramFiles')
addpath('ProgramFiles/TQDM') % Progress bar
addpath('ProgramFiles/AdamarFmincon')
addpath('ProgramFiles/AdamarKmeans')
addpath('ProgramFiles/SPG')
%rng(42);
DATASET = 'Dataset';

small_images = [ 177, 172, 179, 203, 209, 212, 228, 240 ];
descriptors = [Descriptor.Roughness Descriptor.Color];
%descriptors = [Descriptor.Roughness Descriptor.RoughnessGLRL Descriptor.Color];

train_images = 177; %[137, 177, 212, 7, 54, 79];
test_images = [172, 240, 209, 97, 4];
test_images = train_images;

if strcmp(DATASET, 'Dataset2')
    ca = matrix2ca('Dataset2/Descriptors/');
    %ca = matrix2ca('Dataset2/Descriptors512GLRLM/');
    %ca = matrix2ca('Dataset2/DescriptorsProbability/');
    n = numel(ca);
    n_train = floor(n * 0.8);
    n_test = n - n_train;
    X = cell2mat({cell2mat(ca(1:n_train)).X}');
    ca_Y = ca(n_train+1:n);
else
    %X = matfile('X10.mat').X; % descriptors for first 10 images
    %X = cell2mat({cell2mat(matrix2ca('Dataset/SmallImagesDescriptors/')).X}'); % descriptors for "smaller" images
    %X = get_descriptors(load_images(137), descriptors); % hřebík (smallest image)
    X = get_descriptors(load_images(train_images), descriptors);
end

fprintf("How balanced are the labels? Ones: %.2f, Zeros: %.2f\n",...
    sum(X(:,end)), size(X(:,end), 1)-sum(X(:,end)));

if false % Selective MinMaxScaling [-1,1]
    colmin = min(X); % a
    colmax = max(X); % b
    u = 1;
    l = -1;
    cols = colmax > 1; % Select columns to be scaled
    X(:,cols) = l + ...
        ((X(:,cols)-colmin(cols))./(colmax(cols)-colmin(cols))).*(u-l);
end
%X(:,1:end-1) = normalize(X(:,1:end-1));

%ADAMAR
%alpha = [1e-4, 1e-3];
%alpha = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1-1e-1, 1-1e-2, 1-1e-3];
%alpha = 0.01:0.003:0.02; %1e-4*[0.1:0.1:1];
alpha = 1e-6;
K = 10; % Number of clusters
maxIters = 2;

for a = 1:numel(alpha)
    [C, Gamma, PiX, Lambda, it, Lit, learningErrors, stats_train, L] = ...
        adamar_fmincon(X, K, alpha(a), maxIters);
    disp(Lambda);
    Ls(a)=L; % Save objective function value for L-curve

    %EVALUATION
    for i = 1:maxIters
        if i <= it
        lprecision(i) = stats_train(i).precision;
        lrecall(i) = stats_train(i).recall;
        lf1score(i) = stats_train(i).f1score;
        laccuracy(i) = stats_train(i).accuracy;
        else
            lprecision(i) = NaN;
            lrecall(i) = NaN;
            lf1score(i) = NaN;
            laccuracy(i) = NaN;
        end
    end
    
    for i = 1:numel(test_images)
        [stats_test] = adamar_predict(Lambda, C, K, [], [], test_images(i), descriptors);
        %[stats_test] = adamar_predict(Lambda, C, K, colmin, colmax, test_images(i), descriptors);
        %[stats_test] = adamar_predict_mat(Lambda, C, K, [], [], ca_Y, DATASET);
        
        if false % test on training data (combined)
            ca_Y{1}.X = X;
            ca_Y{1}.I = 1;
            [stats_test] = adamar_predict_mat(Lambda, C, K, [], [], ca_Y, DATASET);
        end

        tprecision(i) = stats_test.precision;
        trecall(i) = stats_test.recall;
        tf1score(i) = stats_test.f1score;
        taccuracy(i) = stats_test.accuracy;
    end
    
    %Score plot
    fmincon_score_plot('Adamar fmincon', 1:maxIters, 1:numel(test_images),lprecision, lrecall, lf1score, laccuracy, tprecision, trecall, tf1score, taccuracy)
end

% L-curve
for k=1:length(K)
    figure
    hold on
    title(['K = ' num2str(K)])
    plot([Ls.L1], [Ls.L2],'r.-');
    text(Ls(1).L1,Ls(1).L2,['$\alpha = ' num2str(alpha(1)) '$'],'Interpreter','latex')
    text(Ls(end).L1,Ls(end).L2,['$\alpha = ' num2str(alpha(end)) '$'],'Interpreter','latex')
    hold off
end

figure
subplot(1,3,1)
hold on
plot(alpha,[Ls.L],'r*-')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L$','Interpreter','latex')
subplot(1,3,2)
hold on
plot(alpha,[Ls.L1],'b*-')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L_1$','Interpreter','latex')
subplot(1,3,3)
hold on
plot(alpha,[Ls.L2],'m*-')
xlabel('$\alpha$','Interpreter','latex')
ylabel('$L_2$','Interpreter','latex')
hold off

