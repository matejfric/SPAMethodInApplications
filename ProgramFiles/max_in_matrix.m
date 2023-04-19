function [Ai,Aj,Amax] = max_in_matrix(A)
%MAX_IN_MATRIX

[Amaxs, idx] = max(A);
[Amax, Aj] = max(Amaxs);
Ai = idx(Aj);

end



