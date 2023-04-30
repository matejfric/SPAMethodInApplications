function [X,y] = get_parking_data()
%GET_PARKING_DATA 

[free,full] = load_parking();

[X_free] = compute_lbp_images_stats(free);
[X_full] = compute_lbp_images_stats(full);

X = [X_free; X_full];

%X = scaling(X, [], 'minmax');

y = [ones(size(X_full,1),1); zeros(size(X_free,1),1)];

end

