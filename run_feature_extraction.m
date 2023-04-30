%--------------------------------------------------------------------------
% Feature extraction
%--------------------------------------------------------------------------

clear all
addpath(genpath(pwd));

fprintf("\nFeature extraction in progress... (shouldn't take longer than 5 minutes)\n");

% Define descriptors - which features to extract
descriptors = ["LBP", "LBP_HSV", "LBP_RGB", "StatMomHSV", "StatMomRGB",...
               "GLCM_HSV", "GLCM_RGB", "GLRLM", "GroundTruth"];

extract_features(descriptors);

fprintf("\nFeature extraction finished.\n");