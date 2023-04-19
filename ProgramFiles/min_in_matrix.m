function [Ai,Aj,Amin] = min_in_matrix(A)
%MIN_IN_MATRIX Summary of this function goes here
%   Detailed explanation goes here

[Amins, idx] = min(A);
[Amin, Aj] = min(Amins);
Ai = idx(Aj);

end

