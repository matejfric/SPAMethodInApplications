function [Gamma,it_in] = myspa_compute_Gamma_spg(S,Gamma,X)
%COMPUTE_GAMMA Adamar Gamma problem
%
% X     ...D x T
% S     ...D x K
% Gamma ...K x T

[K,T] = size(Gamma);
Tcoeff = 1;

spgoptions = spgOptions();
spgoptions.maxit = 5e2;
spgoptions.debug = false;
spgoptions.myeps = 1e-8;
spgoptions.alpha_min = 1e-6;
spgoptions.alpha_max = 1e6;

it_in = 0;
for t = 1:T
    
    gamma0 = Gamma(:,t);

    f = @(gamma) f_spg(gamma,X(:,t),S,Tcoeff);
    g = @(gamma) g_spg(gamma,X(:,t),S,Tcoeff);
    p = @(gamma) projection_simplex(gamma);
    
    f_old = f(gamma0);
    [Gamma(:,t),it_in2] = spg(@(gamma) f(gamma),@(gamma) g(gamma),@(gamma) p(gamma),gamma0,spgoptions);
    f_new = f(Gamma(:,t));

    if and(f_new > f_old, abs(f_new - f_old) > 1e-4)
        keyboard
    end

    it_in = max(it_in2,it_in);
end

end

function L = f_spg(gamma,x,S,Tcoeff)
%F_SPG Objective function

L = (1/Tcoeff)*norm(x - S*gamma,2)^2;

end

function g = g_spg(gamma,x,S,Tcoeff)
%G_SPG Gradient function

g = (1/Tcoeff)*(2*(S'*S)*gamma - 2*S'*x);

end

