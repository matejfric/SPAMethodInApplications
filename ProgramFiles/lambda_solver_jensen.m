function [Lambda] = lambda_solver_jensen(piX,piY)
%LAMBDA_SOLVER Solve binary Lambda problem (or after Jensen inequality)

KX = size(piX,1);
KY = size(piY,1);

Lambda = zeros(KY,KX);
for i = 1:KY
    for j = 1:KX
        Lambda(i,j) = dot(piX(j,:),piY(i,:));
    end
end

sumLambda = sum(Lambda,1);
for i=1:KY
    Lambda(i,sumLambda ~= 0) = Lambda(i,sumLambda ~= 0)./sumLambda(sumLambda ~= 0);
end
Lambda(:,sumLambda == 0) = 1/KY;


end

