function [logX] = mylog(X)
%MYLOG Summary of this function goes here
%   Detailed explanation goes here

logX = max(log(X),-1e16);

%logX = log(X);
%logX(X==0) = -1e16;

end

