addpath('Dataset')
addpath('Dataset2')

DATASET = 'Dataset';
COLOR = true;
PROBABILITIES = false;
RESIZE = true;

%images = 0:43;
%images = 1:253;
images = 1:110;
%images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
descriptors = [Descriptor.Roughness Descriptor.Color];
%descriptors = [Descriptor.Roughness Descriptor.Color Descriptor.RoughnessGLRL];
folder = 'Dataset/Descriptors256_binary/';

for i = progress(1:numel(images))
    ca = load_images(images(i), DATASET);
    if RESIZE
		% Why Lanczos resampling? https://stackoverflow.com/questions/384991/what-is-the-best-image-downscaling-algorithm-quality-wise 
        ca{1} = imresize(ca{1},[256 NaN],'lanczos3'); 
        ca{2} = imresize(ca{2},[256 NaN],'lanczos3');
        %figure; imshow(ca{1});
        %figure; imshow(ca{2}, []);
        %imwrite(ca{1},sprintf('%s%d.jpg', 'Dataset/Original256/', images(i)))
        %imwrite(mat2gray(ca{2}),sprintf('%s%d.png', 'Dataset/Annotations256/', images(i)))
    end
    X = get_descriptors(ca, descriptors, COLOR, PROBABILITIES);
    save(sprintf('%sX%d.mat', folder, images(i)),'X');
end