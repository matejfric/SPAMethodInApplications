% Find and save annotations to folder corresponding to original images

addpath(genpath(pwd))

folder='Dataset/Annotations256';
listing = dir('DatasetSelection');

saveFolder = 'SelAnnot';

ca = cell(numel(listing)-2, 1);
for i = 3:numel(listing)
    
    img_num = cell2mat(regexp(listing(i).name,'\d*','Match'));
    filename = [folder  '/'  img_num '.png'];
    img = imread(filename);
    
    imwrite(img,[saveFolder '/' img_num '.png'])

end
