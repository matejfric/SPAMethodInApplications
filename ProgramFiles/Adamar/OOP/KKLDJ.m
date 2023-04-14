classdef KKLDJ < TemplateModel
    %K-means-KLD-Jensen Classifier
    
    methods
        function obj = KKLDJ(K,alpha,maxIt,Nrand,scaleT)
            %KMEANSKLDJENSEN Constructor
            arguments
                K      = 25       % Number of clusters
                alpha  = 0.5;     % Regularization parameter
                maxIt  = 100;     % Maximum number of iterations
                Nrand  = 5;       % Number of random runs
                scaleT = true;    % Scaling of L1
            end
            % Base class constructor
            obj = obj@TemplateModel(K,alpha,maxIt,Nrand,scaleT);
        end
        
        function fit(obj, X_train, y_train)
            % Train the model
            Pi = zeros(2,length(y_train));
            Pi(1,y_train==1) = 1;
            Pi(2,y_train==0) = 1;
            [obj.mdl, obj.L, obj.statsTrain] = train_kkldj(...
                     X_train, Pi, obj.K, obj.alpha, obj.maxIt, obj.Nrand, obj.scaleT, obj.verbose);
        end
        
    end
end



