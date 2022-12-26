images = 0:43;
descriptors = [Descriptor.Roughness Descriptor.Color];
folder = 'Dataset2/Descriptors/';

for i = 1 : numel(images)
    ca = load_images(i-1, 'Dataset2');
    X = get_descriptors(ca, descriptors);
    save(sprintf('%sX%d.mat', folder, i-1),'X');
end