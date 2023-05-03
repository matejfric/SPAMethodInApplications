function [] = sse_elbow_method(X, iters, eps, seed )
%SSE
%   T ... 2D array of 2D points
%   iters ... number of iterations
arguments
    X
    iters = 1e3
    eps = 1e-12
    seed = 42
end
rng(seed)

Ks = 1:10;
sse_final = zeros(numel(Ks),1);
for K = progress(Ks)
    T = size(X,1); % Number of data points in the dataset
    
    [C] = init_random(X, K);
    
    sse = zeros(1,iters); sse(1) = Inf;
    
    for it = 2:iters
        Gamma = zeros(K, T);
        for t = 1:T
            [sse_t,id] = min( sum( (C - X(t,:) ).^2 , 2 ));
            sse(it) = sse(it) + sse_t;
            Gamma(id,t) = 1;
        end
        for k = 1:K
            ids = Gamma(k,:) == 1;
            C(k,:) = mean(X(ids,:));
        end
        
        % Terminating condition
        if norm(sse(it-1) - sse(it)) < eps
            sse_final(K) = sse(it);
            break
        end
    end
end

figure
plot(Ks, sse_final(Ks),'-o', 'LineWidth', 2);
xlabel('$K$','Interpreter','latex','FontSize',14);
ylabel('SSE','Interpreter','latex','FontSize',14);
xticks(Ks);
%title('SSE','FontSize', 14);
ax = gca;
ax.FontSize = 12; 
grid on
grid minor

set(gca, 'YScale', 'log')

end

