function [X,y] = get_ionosphere_data()
%GET_IONOSPHERE_DATA 

load ionosphere
X(:,2) = [];
y = zeros(length(Y),1);
y([Y{:}]=='g') = 1;

end

