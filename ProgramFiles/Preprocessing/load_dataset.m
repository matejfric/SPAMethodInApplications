function [ca_dataset] = load_dataset(number_of_images, dataset_folder,images_subfolder, annotations_subfolder)
%LOAD_DATASET Loads the dataset.

switch nargin
        case 0
            number_of_images = numel(dir(sprintf('%s/%s/*.jpg',...
                dataset_folder, images_subfolder))); % Number of images in the dataset
        case 1
            dataset_folder = 'Dataset';
            images_subfolder = 'Original';
            annotations_subfolder = 'Annotations';
        case 2
            images_subfolder = 'Original';
            annotations_subfolder = 'Annotations';
end

fprintf("Loading dataset...\n");
ca_dataset = cell(number_of_images,2);

for k = progress(1:number_of_images)
  jpgFile = strcat(num2str(k), '.jpg');
  pngAnnotFile = strcat(num2str(k), '.png');
  fullJpgFile = fullfile(dataset_folder, images_subfolder, jpgFile);
  fullPngAnnotFile = fullfile(dataset_folder, annotations_subfolder, pngAnnotFile);
  % Original image:
  if exist(fullJpgFile, 'file')
    ca_dataset{k,1} = imread(fullJpgFile);
  else
    warningMessage = sprintf('Warning: image file does not exist:\n%s', fullJpgFile);
    uiwait(warndlg(warningMessage));
  end
  %Annotated image:
  if exist(fullPngAnnotFile, 'file')
    ca_dataset{k,2} = imread(fullPngAnnotFile);
  else
    warningMessage = sprintf('Warning: image file does not exist:\n%s', fullPngAnnotFile);
    uiwait(warndlg(warningMessage));
  end
end
fprintf("The dataset has been successfully loaded.\n");

end

