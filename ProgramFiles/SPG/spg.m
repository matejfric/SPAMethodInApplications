function [x,it,perf] = spg(f_func,g_func,P_func,x0,options)
%SPG Summary of this function goes here
%   Detailed explanation goes here

x = P_func(x0);
g = g_func(x);
g(isinf(g)) = 1e2;

it = 0;
it_in_all = 0;
f_it = [];
d_it = [];
alpha_bb = options.alpha_bb_init;%0.5*(options.alpha_min + 1/options.alpha_min);

f = f_func(x);
f_old = Inf;
fm = f*ones(1,options.M);

while it < options.maxit
    x_old = x;
    g_old = g;
    
    d = P_func(x - alpha_bb*g) - x;
    [beta, it_in] = compute_step_size_gll(f_func,g_func,max(fm),d,x,options);

    if abs(sum(P_func(x - alpha_bb*g)) - 1) > 1e-8
%        keyboard
    end
    
    x = x + beta*d;
    g = g_func(x);
%    g(isinf(g)) = 1e2;
    
    s = x - x_old;
    y = g - g_old;
    
    ss = dot(s,s);
    sy = dot(s,y);
%    ss = dot(s(~isinf(y)),s(~isinf(y)));
%    sy = dot(s(~isinf(y)),y(~isinf(y)));

%    ss = dot(s,g);
%    sy = dot(d,y);

    if sy <= 0
        alpha_bb = 1/options.alpha_min;
    else
        alpha_bb = min([max([options.alpha_min,ss/sy]),1/options.alpha_min]);
    end
    
    f_old = f;
    f = f_func(x);
    fm = [fm(2:end),f];
    
    d_tilde = P_func(x - options.alpha_bb_init*g) - x;
    if options.debug
        disp([num2str(it) ': f = ' num2str(f) ', norm_dtilde = ' num2str(norm(d_tilde,2)) ', norm_d = ' num2str(norm(d,2)) ', it_in = ' num2str(it_in) ', alpha_bb = ' num2str(alpha_bb) ', beta = ' num2str(beta)])
    end
    
    it_in_all = it_in_all + it_in;
    it = it + 1;

    f_it(it) = f;

    d_it(it) = norm(d_tilde,2);

    %    if or(norm(d_tilde,2) < options.myeps, beta < options.myeps)
%    if norm(d_tilde,2) < options.myeps
    if norm(d,2) < options.myeps
%    if abs(f - f_old) < options.myeps
        break;
    end

end

perf.f_it = f_it;
perf.d_it = d_it;
perf.it_in_all = it_in_all;

if options.debug
%if it > 900
    figure
    hold on
    title('SPG performance')
    subplot(1,2,1)
    plot(f_it,'b')
    xlabel('it')
    ylabel('f')
    hold off
    subplot(1,2,2)
    plot(d_it,'r')
    xlabel('it')
    ylabel('norm(d)')
    set(gca,'yscale','log')
    hold off
    
%    close all
end

end

% compute stepsize using GLL
function [beta, it_in] = compute_step_size_gll(F_func,G_func,f_max,d,X,options)
it_in = 1;

beta = 1;
X_temp = X + beta*d;
G = G_func(X);
delta = dot(G,d);

f = F_func(X);
f_X_temp = F_func(X_temp);

while and(f_X_temp > f_max + options.gamma*beta*delta, it_in < 500)
    beta_temp = - 0.5*(beta^2*delta)/(f_X_temp - f - beta*delta);
    if and(beta_temp >= options.sigma1, beta_temp <= options.sigma2*beta)
        beta = beta_temp;
    else
        beta = options.c*beta;
    end
    X_temp = X + beta*d;
    
    f_X_temp = F_func(X_temp);
    
    it_in = it_in + 1;
    
end
end

