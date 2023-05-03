function [L] = myspa_compute_L(S,Gamma,X,Tcoeff)
%SPA_COMPUTE_L Compute objective function value for SPA

L = (1/Tcoeff)*norm(X - S*Gamma,'fro')^2;

end


