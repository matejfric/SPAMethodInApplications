classdef spgOptions
    %SPG_OPTIONS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        maxit
        minit
        myeps
        alpha_bb_init
        gamma
        sigma1
        sigma2
        alpha_min
        alpha_max
        beta_max
        M
        
        debug
    end
    
    methods
        function obj = spgOptions()
            % constructor - set default values
            obj.maxit = 1e2;
            obj.minit = -1;
            obj.myeps = 1e-5;
            obj.alpha_bb_init = 1e-3;
            obj.gamma = 0.9;
            obj.sigma1 = 1e-4;
            obj.sigma2 = 1 - obj.sigma1;
            obj.alpha_min = 1e-4;
            obj.alpha_max = 1e4;
            obj.M = 12;
            obj.beta_max = 0.95;
            
            obj.debug = false;
        end
        
    end
end    