function [X,y] = get_bcwd_data()
%GET_BCWD_DATA

warning('off','MATLAB:table:ModifiedAndSavedVarnames');
DS = readtable('bcwd.csv');
DS(:,1) = []; % drop ID
X = table2array(DS(:,2:end));
y = zeros(length(X),1);
labels=table2array(DS(:,1));
y([labels{:}]=='M')=1;
%debug: [array2table(y), DS(:,1)]

end

