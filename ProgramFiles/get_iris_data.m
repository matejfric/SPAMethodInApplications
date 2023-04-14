function [X,y] = get_iris_data()
%GET_IRIS_DATA 

load fisheriris;
X = meas;

species = categorical(species);
classes = categories(species);
Pi = onehotencode(species,2);
y = onehotdecode(Pi,[1,2,3],2);
y = double(y);

end

