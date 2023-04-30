function [X,it] = spg_vec(F_func,G_func,P_func,X0,options)
%SPG Summary of this function goes here
%   Detailed explanation goes here

T = size(X0,2);

X = P_func(X0);
G = G_func(X);

it = 0;
f_it = [];
alpha_bb = options.alpha_bb_init*ones(1,T);

f = F_func(X);

fm = kron(ones(options.M,1),f);

while it < options.maxit
    X_old = X;
    G_old = G;
    
    d = P_func(X - bsxfun(@times,G,alpha_bb)) - X;
    
    max_fm = max(fm,[],1);
    [beta, it_in] = compute_step_size_gll(F_func,G_func,max_fm,d,X,options.gamma,options.sigma1,options.sigma2,options.beta_max);
    
    X = X + bsxfun(@times,d,beta);
    G = G_func(X);
    
    S = X - X_old;
    Y = G - G_old;
    
    SS = sum(S.^2,1);
    SY = sum(S.*Y,1);
    
    cond1 = (SY <= 0);
    alpha_bb(cond1) = options.alpha_max;
    alpha_bb(~cond1) = min(max(SS(~cond1)./SY(~cond1),options.alpha_min),options.alpha_max);
    
    f = F_func(X);
    fm = [fm(2:end,:);f];
    
    if options.debug
        disp([num2str(it) ': f = ' num2str(sum(f)) ', norm_d = ' num2str(norm(d,'fro')) ', it_in = ' num2str(it_in) ', alpha_bb = ' num2str(min(alpha_bb)) ', beta = ' num2str(min(beta))])
    end
    
    if and(max(sum(d.^2,1)) < options.myeps^2, it > options.minit)
        break;
    end
    it = it + 1;
    
    f_it(it) = sum(f);
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

T = size(X,2);

beta = beta_max*ones(1,T);
X_temp = X + bsxfun(@times,d,beta);
G = G_func(X);

delta = sum(G.*d,1);

f = F_func(X);
f_X_temp = F_func(X_temp);

solved = zeros(1,T);

it_in = 1;
while and(sum(solved) < T,it_in < 500)
    cond0 = (f_X_temp <= f_max + gamma*beta.*delta);
    solved(cond0) = 1;
    
%    beta_temp = - 0.5*(beta(solved == 0).^2.*delta(solved == 0))./(f_X_temp(solved == 0) - f(solved == 0) - beta(solved == 0).*delta(solved == 0));
    beta_temp = - 0.5*((beta.^2).*delta)./(f_X_temp - f - beta.*delta);

    cond1 = and(solved == 0, and(beta_temp >= sigma1, beta_temp <= sigma2*beta));
    cond2 = and(solved == 0, ~and(beta_temp >= sigma1, beta_temp <= sigma2*beta));
    beta(cond1) = beta_temp(cond1);
    beta(cond2) = beta(cond2)/2;
    X_temp = X + bsxfun(@times,d,beta);
    f_X_temp = F_func(X_temp);
    
    it_in = it_in + 1;
end

end

