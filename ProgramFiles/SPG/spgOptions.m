classdef spgOptions
    %SPG_OPTIONS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        maxit
        myeps
        alpha_bb_init
        
        c
        gamma
        sigma1
        sigma2
        alpha_min
        M
        
        debug
    end
    
    methods
        function obj = spgOptions()
            % constructor - set default values
            obj.maxit = 1e2;
            obj.myeps = 1e-4;
            obj.alpha_bb_init = 1;

            obj.c = 0.5;
            obj.gamma = 0.9;
            obj.sigma1 = 1e-4;
            obj.sigma2 = 1 - obj.sigma1;
            obj.alpha_min = 1e-3;
            obj.M = 12;

            obj.debug = false;
        end
        
    end
end    