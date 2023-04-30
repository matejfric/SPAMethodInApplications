%--------------------------------------------------------------------------
% Feature extraction
%--------------------------------------------------------------------------

clear all
addpath(genpath(pwd));

fprintf("\nFeature extraction in progress..."+...
        "(shouldn't take longer than 5 minutes)\n");

% Define descriptors - which features to extract
descriptors = ["GroundTruth", "StatMomHSV", "StatMomRGB",...
               "GLCM_HSV", "GLCM_RGB", "GLCM_Gray"];

extract_features(descriptors);

fprintf("\nFeature extraction finished successfully.\n");