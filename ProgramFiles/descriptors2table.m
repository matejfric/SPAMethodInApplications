function [tbl] = descriptors2table(X,y)
%DESCRIPTORS2TABLE

descriptors = ["LBP", "LBP_HSV", "LBP_RGB", "StatMomHSV", "StatMomRGB",...
               "GLCM_HSV", "GLCM_RGB", "GLRLM"];


% Split the LBP column into 6 sub-columns (LBP1 to LBP6)
LBP = array2table(X(:, 1:6));
LBP.Properties.VariableNames = {'LBP1', 'LBP2', 'LBP3', 'LBP4', 'LBP5', 'LBP6'};

% Split the LBP_HSV column into 18 sub-columns 
%(LBPH1 to LBPH6, LBPS1 to LBPS6, LBPV1 to LBPV6)
LBP_HSV = array2table(X(:, 7:24));
LBP_HSV_vars = cell(1, 18);
for i = 1:6
    LBP_HSV_vars{i} = sprintf('LBP_H%d', i);
    LBP_HSV_vars{i+6} = sprintf('LBP_S%d', i);
    LBP_HSV_vars{i+12} = sprintf('LBP_V%d', i);
end
LBP_HSV.Properties.VariableNames = LBP_HSV_vars;
                                
% Split the LBP_RGB column into 18 sub-columns 
% (LBP_R1 to LBP_R6, LBP_G1 to LBP_G6, LBP_B1 to LBP_B6)
LBP_RGB = array2table(X(:, 25:42));
LBP_RGB_vars = cell(1, 18);
for i = 1:6
    LBP_RGB_vars{i} = sprintf('LBP_R%d', i);
    LBP_RGB_vars{i+6} = sprintf('LBP_G%d', i);
    LBP_RGB_vars{i+12} = sprintf('LBP_B%d', i);
end
LBP_RGB.Properties.VariableNames = LBP_RGB_vars;

% Split the StatMomHSV column into 18 sub-columns for H, S, and V 
% (SMH1 to SMH6, SMS1 to SMS6, SMV1 to SMV6)
StatMomHSV = array2table(X(:, 43:60));
StatMomHSV_vars = cell(1, 18);
for i = 1:6
    StatMomHSV_vars{i} = sprintf('SMH%d', i);
    StatMomHSV_vars{i+6} = sprintf('SMS%d', i);
    StatMomHSV_vars{i+12} = sprintf('SMV%d', i);
end
StatMomHSV.Properties.VariableNames = StatMomHSV_vars;

% Split the StatMomRGB column into 18 sub-columns for R, G, and B 
% (SMR1 to SMR6, SMG1 to SMG6, SMB1 to SMB6)
StatMomRGB = array2table(X(:, 61:78));
StatMomRGB_vars = cell(1, 18);
for i = 1:6
    StatMomRGB_vars{i} = sprintf('SMR%d', i);
    StatMomRGB_vars{i+6} = sprintf('SMG%d', i);
    StatMomRGB_vars{i+12} = sprintf('SMB%d', i);
end
StatMomRGB.Properties.VariableNames = StatMomRGB_vars;
                                
% Split the GLCM_HSV column into 16 sub-columns per color channel
% (GLCMH1 to GLCMH16, GLCMS1 to GLCMS16, GLCMV1 to GLCMV16)
GLCM_HSV = array2table(X(:, 79:126));
GLCM_HSV_vars = cell(1, 48);
for i = 1:16
    GLCM_HSV_vars{i} = sprintf('GLCMH%d', i);
    GLCM_HSV_vars{i+16} = sprintf('GLCMS%d', i);
    GLCM_HSV_vars{i+32} = sprintf('GLCMV%d', i);
end
GLCM_HSV.Properties.VariableNames = GLCM_HSV_vars;

% Split the GLCM_RGB column into 16 sub-columns per color channel
% (GLCMR1 to GLCMR16, GLCMG1 to GLCMG16, GLCMB1 to GLCMB16)
GLCM_RGB = array2table(X(:, 127:174));
GLCM_RGB_vars = cell(1, 48);
for i = 1:16
    GLCM_RGB_vars{i} = sprintf('GLCMR%d', i);
    GLCM_RGB_vars{i+16} = sprintf('GLCMG%d', i);
    GLCM_RGB_vars{i+32} = sprintf('GLCMB%d', i);
end
GLCM_RGB.Properties.VariableNames = GLCM_RGB_vars;

% GLRLM
GLRLM = array2table(X(:, 175:end));
GLRLM_names = cell(1, 44);
for i = 1:44
    GLRLM_names{i} = sprintf('GLRLM%d', i);
end
GLRLM.Properties.VariableNames = GLRLM_names;

GT = array2table(y);
GT.Properties.VariableNames = {'GT'};

% Concatenate the tables
tbl = horzcat(LBP, LBP_HSV, LBP_RGB, StatMomHSV, StatMomRGB, GLCM_HSV, GLCM_RGB, GLRLM, GT);

end

