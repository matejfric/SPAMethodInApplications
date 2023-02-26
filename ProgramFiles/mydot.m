function [fx] = mydot(A,B)
%MYDOT Summary of this function goes here
%   Detailed explanation goes here

if size(A,2) > 1
    fx = sum(A.*B);
else
    fx = dot(A,B);
end

end

