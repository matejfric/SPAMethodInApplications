classdef KKLD < TemplateModel
    % K-means and KLD (numerical)
    methods
        function obj = KKLD(K,epsilon,maxIt,Nrand,scaleT)
            %KKLD Constructor
            arguments
                K      = 25       % Number of clusters
                epsilon  = 0.5;     % Regularization parameter
                maxIt  = 100;     % Maximum number of iterations
                Nrand  = 5;       % Number of random runs
                scaleT = true;    % Scaling of L1
            end
            % Base class constructor
            obj = obj@TemplateModel(K,epsilon,maxIt,Nrand,scaleT);
        end
        
        function fit(obj, X_train, y_train)
            % Train the model
            [Pi, classes] = myonehotencode(y_train);
            [obj.mdl, obj.L, obj.statsTrain] = train_kkld(...
                     X_train, Pi, obj.K, obj.epsilon, obj.maxIt, obj.Nrand, obj.scaleT, obj.verbose);
            obj.mdl.classes = classes;
        end
        
    end
end

