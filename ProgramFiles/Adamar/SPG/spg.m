function [X,it] = spg(F_func,G_func,P_func,X0,options)
%SPG Summary of this function goes here
%   Detailed explanation goes here

X = P_func(X0);
G = G_func(X);

it = 0;
f_it = [];
alpha_bb = options.alpha_bb_init;

f = F_func(X);
fm = f*ones(1,options.M);

while it < options.maxit
    X_old = X;
    G_old = G;
    
    d = P_func(X - alpha_bb*G) - X;
    [beta, it_in] = compute_step_size_gll(F_func,G_func,max(fm),d,X,options.gamma,options.sigma1,options.sigma2,options.beta_max);
    
    X = X + beta*d;
    G = G_func(X);
   
    S = X - X_old;
    Y = G - G_old;
    
    SS = mydot(S,S);
    SY = mydot(S,Y);
    if SY <= 0
        alpha_bb = options.alpha_max;
    else
        alpha_bb = min([max([options.alpha_min,SS/dot(S,Y)]),options.alpha_max]);
    end
    
    f = F_func(X);
    fm = [fm(2:end),f];
    
    if options.debug
        disp([num2str(it) ': f = ' num2str(f) ', norm_d = ' num2str(norm(d,2)) ', it_in = ' num2str(it_in) ', alpha_bb = ' num2str(alpha_bb) ', beta = ' num2str(beta)])
    end
    
    if and(norm(d,'fro') < options.myeps, it > options.minit)
        break;
    end
    it = it + 1;

    f_it(it) = f;
end

if options.debug
    figure
    hold on
    plot(f_it)
    hold off
end

end

% compute stepsize using GLL
function [beta, it_in] = compute_step_size_gll(F_func,G_func,f_max,d,X,gamma,sigma1,sigma2,beta_max)
it_in = 1;

beta = beta_max;
X_temp = X + beta*d;
G = G_func(X);
delta = mydot(G,d);

f = F_func(X);
f_X_temp = F_func(X_temp);

while and(f_X_temp > f_max + gamma*beta*delta, it_in < 500)
    beta_temp = - 0.5*(beta^2*delta)/(f_X_temp - f - beta*delta);
    if and(beta_temp >= sigma1, beta_temp <= sigma2*beta)
        beta = beta_temp;
    else
        beta = beta/2;
    end
    X_temp = X + beta*d;
    
    f_X_temp = F_func(X_temp);
    
    it_in = it_in + 1;
    
end
end
