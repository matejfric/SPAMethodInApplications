function [I] = myim2gray(Irgb)
%MYIM2GRAY Summary of this function goes here
%   Detailed explanation goes here

I = mean(Irgb,3);

end

