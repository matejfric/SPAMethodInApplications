classdef KLambda < TemplateModel
    %K-means & Lambda solver (assumes non-conflicting objectives)
    methods
        function obj = KLambda(K,epsilon,maxIt,Nrand,scaleT)
            %KLambda Constructor
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
            [obj.mdl, obj.L, obj.statsTrain] = train_klambda(...
                X_train, Pi, obj.K, obj.epsilon);
            obj.mdl.classes = classes;
        end
        
    end
end

