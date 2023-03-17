addpath(genpath(pwd))

DATASET = 'Dataset';
PROBABILITIES = false;
RESIZE = true;

folder='DatasetSelection';
originalFolder = 'Original';
annotationsFolder = 'Annotations';
listing = dir([folder '/' originalFolder]);
ca = cell(numel(listing)-2, 3);
for i = 3:numel(listing)
    img_num = cell2mat(regexp(listing(i).name,'\d*','Match'));
    
    orig_img = imread([folder  '/'  originalFolder '/' img_num '.jpg']);
    anno_img = imread([folder  '/'  annotationsFolder '/' img_num '.png']);
    
    ca{i-2,1} = orig_img;
    ca{i-2,2} = anno_img;
    ca{i-2,3} = img_num;
    
%     figure; imshow(ca{i-2,1});
%     figure; imshow(ca{i-2,2}, []);
end

descriptors = [Descriptor.StatMomHSV];

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
    X = get_descriptors(ca(i,1:2), descriptors, PROBABILITIES);
    %TODO: X = get_GLCM_pixel_value
    save(sprintf('%sX%s.mat', [folder '/Descriptors/StatMomHSV34/'], ca{i,3}),'X');
end