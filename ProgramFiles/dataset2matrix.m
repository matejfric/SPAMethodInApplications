images = 0:43;
%images = [ 172, 177, 179, 203, 209, 212, 228, 240 ];
descriptors = [Descriptor.Roughness Descriptor.Color];
folder = 'Dataset2/DescriptorsProbability/';

for i = 1 : numel(images)
    ca = load_images(images(i), 'Dataset2');
    X = get_descriptors(ca, descriptors);
    save(sprintf('%sX%d.mat', folder, images(i)),'X');
end