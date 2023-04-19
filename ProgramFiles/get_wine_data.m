function [X,y] = get_wine_data()
%GET_WINE_DATA 
warning('off','MATLAB:table:ModifiedAndSavedVarnames');
DS = readtable('winequality-red.csv');

X = table2array(DS(:,1:end-1));

% Transformation of the target variable for binary classification
y = discretize(DS.quality,[2, 5.5, 8]) - 1; 

% Multiclass
%y = DS.quality; 


end

