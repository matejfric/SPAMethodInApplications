function [ca_images] = load_images_from_folder(folder_path)

    if ~exist(folder_path, 'dir')
        error('Folder does not exist.');
    end
    
    listing = dir(fullfile(folder_path,'*.png'));
    n_files = length(listing);
    ca_images = cell(n_files,1);
    
    for i = 1:n_files
        filename = fullfile(listing(i).folder, listing(i).name);
        img = imread(filename);
        M.X = img;
        M.I = listing(i).name;
        ca_images{i} = M;
    end
end

