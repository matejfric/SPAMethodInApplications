function [] = extract_features(descriptors_str)
%EXTRACT_FEATURES 

descriptors_enum = map_descriptors(descriptors_str);
PROBABILITIES = false;
RESIZE = false;
folder='DatasetSelection';

original_folder = 'Original256';
annotations_folder = 'Annotations256';

listing = dir([folder '/' original_folder]);
ca = cell(numel(listing)-2, 3);
for i = 3:numel(listing)
    img_num = cell2mat(regexp(listing(i).name,'\d*','Match'));
    
    orig_img = imread([folder  '/'  original_folder '/' img_num '.jpg']);
    anno_img = imread([folder  '/'  annotations_folder '/' img_num '.png']);
    
    ca{i-2,1} = orig_img;
    ca{i-2,2} = anno_img;
    ca{i-2,3} = img_num;
    
%     figure; imshow(ca{i-2,1});
%     figure; imshow(ca{i-2,2}, []);
end

descriptors_folder = '/Descriptors';

if ~exist([folder descriptors_folder], 'dir')
    mkdir([folder descriptors_folder]);
end

for d = 1:length(descriptors_str)
    
    desc_enum = descriptors_enum(d);
    folder_name = descriptors_str(d);
    full_folder = strjoin([folder descriptors_folder '/' folder_name '/'],"");

    if ~exist(full_folder, 'dir')
        mkdir(full_folder);
        addpath(full_folder);
    end


    n = size(ca, 1);
    for i = progress(1:n)
        if RESIZE
            % Why Lanczos resampling? https://stackoverflow.com/questions/384991/what-is-the-best-image-downscaling-algorithm-quality-wise 
            ca{i,1} = imresize(ca{i,1},[256 NaN],'lanczos3'); 
            ca{i,2} = imresize(ca{i,2},[256 NaN],'lanczos3');
    %         figure; imshow(ca{i,1});
    %         figure; imshow(ca{i,2}, []);

    %         imwrite(ca{i,1},sprintf('%s%s.jpg', [folder '/Original256/'], ca{i,3}))
    %         imwrite(mat2gray(ca{i,2}),sprintf('%s%s.png', [folder '/Annotations256/'], ca{i,3}))
        end
        X = get_descriptors(ca(i,1:2), desc_enum, PROBABILITIES);
        save(sprintf('%sX%s.mat', full_folder, ca{i,3}),'X');
    end
end

end

