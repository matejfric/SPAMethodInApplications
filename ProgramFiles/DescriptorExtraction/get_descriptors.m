function [X] = get_descriptors(ca, descriptors, probability, offsets)
%GET_DESCRIPTORS Performs roughness and color analysis
% X...Matrix of descriptors
arguments
    ca cell
    descriptors = [ Descriptor.Color, Descriptor.Roughness]
    probability = false;
    offsets = [0 1;-1 1;-1 0;-1  -1];
end
if isempty(descriptors)
    descriptors = [ Descriptor.Color, Descriptor.Roughness];
end
if ismember(Descriptor.Color, descriptors)
    descriptors = [descriptors, Descriptor.StatMomHSV, Descriptor.StatMomRGB];
end
if ismember(Descriptor.Roughness, descriptors)
    descriptors = [descriptors, Descriptor.GLCM_Gray, Descriptor.GLCM_RGB, Descriptor.GLCM_HSV];
end

%Statistical moments HSV
if ismember(Descriptor.StatMomHSV, descriptors)
    X_SM_HSV = color_analysis_hsv(ca);
else
    X_SM_HSV = [];
end

%Statistical moments RGB
if ismember(Descriptor.StatMomRGB, descriptors)
    X_SM_RGB = color_analysis_rgb(ca);
else
    X_SM_RGB = [];
end

%Gray level co-occurrence matrix
if ismember(Descriptor.GLCM_Gray, descriptors)
    X_GLCM_Gray = roughness_analysis_glcm_gray(ca, offsets);
else
    X_GLCM_Gray = [];    
end

if ismember(Descriptor.GLCM_RGB, descriptors)
    X_GLCM_RGB = roughness_analysis_glcm_rgb(ca, offsets);
else
    X_GLCM_RGB = [];    
end

if ismember(Descriptor.GLCM_HSV, descriptors)
    X_GLCM_HSV = roughness_analysis_glcm_hsv(ca, offsets);
else
    X_GLCM_HSV = [];    
end

% Grey-level run-length matrix
if ismember(Descriptor.GLRLM, descriptors)
    X_GLRLM = roughness_analysis_glrl(ca);
else
    X_GLRLM = [];    
end

% Local Binary Patterns
if ismember(Descriptor.LBP, descriptors)
    X_LBP = lbp_analysis(ca);
else
    X_LBP = [];    
end

% Ground truth
if ismember(Descriptor.GroundTruth, descriptors)
    X_True = get_ground_truth(ca, probability);
else
    X_True = [];
end

% Matrix of descriptors
X = [X_GLCM_Gray, X_GLCM_RGB, X_GLCM_HSV, X_GLRLM,...
     X_SM_RGB, X_SM_HSV, X_LBP, X_True];

end

