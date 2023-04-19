function [free,full] = load_parking()
%LOAD_PARKING

free = load_images_from_folder(fullfile('DatasetParking','free'));
full = load_images_from_folder(fullfile('DatasetParking','full'));

end



