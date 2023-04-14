function [descriptorsEnum] = map_descriptors(descriptors)
%MAPDESCRIPTORS 

mapping =  {'LBP', Descriptor.LBP;
            'LBP_HSV', Descriptor.LBP_HSV;
            'LBP_RGB', Descriptor.LBP_RGB;
            'StatMomHSV', Descriptor.StatMomHSV;
            'StatMomRGB', Descriptor.StatMomRGB;
            'GLCM_HSV', Descriptor.GLCM_HSV;
            'GLCM_RGB', Descriptor.GLCM_RGB;
            'GLRLM', Descriptor.GLRLM;
            'Color', Descriptor.Color;
            'Roughness', Descriptor.Roughness;
            'GLCM_Gray', Descriptor.GLCM_Gray;
            'GroundTruth', Descriptor.GroundTruth;
            'LBP_RGB', Descriptor.LBP_RGB;
            'LBP_HSV', Descriptor.LBP_HSV;
            };

for i = 1:numel(descriptors)
    idx = find(strcmp(mapping(:,1), descriptors(i)));
    if ~isempty(idx)
        descriptorsEnum(i) = mapping{idx, 2};
    end
end

end