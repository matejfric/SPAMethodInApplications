function [Ai,Aj,Amax] = max_in_matrix(A)
%MIN_IN_MATRIX Summary of this function goes here
%   Detailed explanation goes here

[Amaxs, idx] = max(A);
[Amax, Aj] = max(Amaxs);
Ai = idx(Aj);

end

