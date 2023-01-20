function PX = projection_simplex(X)

% projection onto simplexes - vectorized variant
[K,T] = size(X);
Y = sort(X,1);

t_hat = zeros(1,T,class(X));
i = (K-1);
I = i*ones(1,T,class(X)); % store i for which simplex was solved

sumI = T; % number of unresolved simplexes
while sumI > 0 && i >= 1
    t_hat(I == i) = (sum(Y(i+1:K,I == i),1)-1)/(K-i);
    I(t_hat >= Y(i,:) & I == i) = -1;
    I(I == i) = i-1;

    i = i - 1;
    sumI = sum(I == i);
end

t_hat(I == 0) = (sum(Y(:,I == 0),1)-1)/K;

PX = max(X - kron(ones(K,1,class(X)),t_hat),0);

end