function [X] = get_descriptors(ca, descriptors, COLOR, PROBS)
%GET_DESCRIPTORS Performs roughness and color analysis
arguments
    ca cell
    descriptors = [ Descriptor.Color, Descriptor.Roughness]
    COLOR = false;
    PROBS = false;
end
if isempty(descriptors)
    descriptors = [ Descriptor.Color, Descriptor.Roughness];
end

if ismember(Descriptor.Color, descriptors)
    X_Color = color_analysis(ca);
else
    X_Color = [];
end

if ismember(Descriptor.Roughness, descriptors)
    X_GLCM = roughness_analysis(ca, COLOR);
else
    X_GLCM = [];    
end

if ismember(Descriptor.RoughnessGLRL, descriptors)
    X_GLRLM = roughness_analysis_glrl(ca);
else
    X_GLRLM = [];    
end
  
% X_GLCM = roughness_analysis(ca);
% X_GLRLM = roughness_analysis_glrl(ca);
% X_Color = color_analysis(ca);
X_True = get_ground_truth(ca, PROBS);

X = [X_GLCM, X_GLRLM, X_Color, X_True];

%Normalization?
%X(:,1:end-1) = normalize(X(:,1:end-1));

end

