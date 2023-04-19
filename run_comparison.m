clear all
addpath(genpath(pwd));
rng(13);

%[X,y] = get_iris_data();
%[X,y] = get_bcwd_data();
[X,y] = get_wine_data();
%[X,y] = get_ionosphere_data();
X = scaling(X, [], 'minmax');
%X = scaling(X, [], 'zscore', 'robust');

kfolds = 5;
[X_train, X_test, y_train, y_test] = kfold_crossval(X, y, kfolds);

classifiers{1} = KLambda();
classifiers{2} = KKLDJ();
classifiers{3} = KKLD();
%classifiers{4} = SPAKLD();
%clf_names = pad(["KLambda", "KKLDJ", "KKLD", "SPAKLD", "KNN", "SVM", "D-Tree"],'both');
clf_names = pad(["KLambda", "KKLDJ", "KKLD", "KNN", "Linear SVM", "D-Tree"],'both');

Ks = 3;
epsilons = 10.^(-6:1:6);

for kfold=progress(1:kfolds)
    for clf=1:length(classifiers)
        for a = 1:length(epsilons)
            for k = 1:length(Ks)
                classifiers{clf}.epsilon = epsilons(a);
                classifiers{clf}.K = Ks(k);
                classifiers{clf}.fit(X_train{kfold},y_train{kfold})

%                 Ls{clf}(kfold,a,k)  = classifiers{clf}.L.L;
%                 L1s{clf}(kfold,a,k) = classifiers{clf}.L.L1;
%                 L2s{clf}(kfold,a,k) = classifiers{clf}.L.L2;

                [~,y_pred] = classifiers{clf}.predict(X_test{kfold}');
                stats_test{clf}(kfold,a,k)  = classifiers{clf}.computeStats(y_pred, y_test{kfold}');
                stats_train{clf}(kfold,a,k) = classifiers{clf}.statsTrain;
            end
        end
    end
    % KNN
    clf = length(classifiers) + 1;
    mdlKNN = fitcknn(X_train{kfold}, y_train{kfold}, 'NumNeighbors', Ks);
    x_pred = mdlKNN.predict(X_train{kfold});
    stats_train{clf}(kfold,1) = statistics_multiclass(x_pred, y_train{kfold});
    y_pred = mdlKNN.predict(X_test{kfold});
    stats_test{clf}(kfold,1) = statistics_multiclass(y_pred, y_test{kfold});

    % SVM
    clf = length(classifiers) + 2;
    if sum(unique(y_train{kfold}))<=2
        mdlSVM = fitcsvm(X_train{kfold}, y_train{kfold});
    else
        mdlSVM = fitcecoc(X_train{kfold}, y_train{kfold});
    end
    x_pred = mdlSVM.predict(X_train{kfold});
    stats_train{clf}(kfold,1) = statistics_multiclass(x_pred, y_train{kfold});
    y_pred = mdlSVM.predict(X_test{kfold});
    stats_test{clf}(kfold,1) = statistics_multiclass(y_pred, y_test{kfold});

    % Decision Tree
    clf = length(classifiers) + 3;
    mdlTree = fitctree(X_train{kfold}, y_train{kfold});
    x_pred = mdlTree.predict(X_train{kfold});
    stats_train{clf}(kfold,1) = statistics_multiclass(x_pred, y_train{kfold});
    y_pred = mdlTree.predict(X_test{kfold});
    stats_test{clf}(kfold,1) = statistics_multiclass(y_pred, y_test{kfold});
end


for clf=1:length(classifiers)+3
    f1s{clf} = mean(reshape([stats_test{clf}.f1score],size(stats_test{clf})),1);
    acc{clf} = mean(reshape([stats_test{clf}.accuracy],size(stats_test{clf})),1);
    lf1s{clf} = mean(reshape([stats_train{clf}.f1score],size(stats_train{clf})),1);
    lacc{clf} = mean(reshape([stats_train{clf}.accuracy],size(stats_train{clf})),1);
end


fprintf("|                   Testing            |          Training            |\n");
for clf=1:length(classifiers)+3
    fprintf("| %s: F1 = %.3f  |  ACC = %.3f  |  F1 = %.3f  |  ACC = %.3f  |\n",...
        clf_names(clf), max([f1s{clf}]), max([acc{clf}]),max([lf1s{clf}]), max([lacc{clf}]))
end

