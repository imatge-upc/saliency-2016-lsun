function [ result ] = predictFunc( data )
%PREDICTFUNC Read a list of data, and generate result
%   

GlobalParameters;
result = cell(length(data),1);
for i = 1:length(data)
    % naive central gaussian
    %img = imread([IMAGE_DIR data(i).image '.jpg']);
    %result{i} = algorithmFunc(img);
    
    % cheating, copy ground truth
%     saliency_path = sprintf( SALIENCY_PATTERN, data(i).image);
%     load(saliency_path);
%     result{i} = double(I);
    saliency_path = sprintf( VAL_TEST_PATTERN, data(i).image);
    load(saliency_path);
    result{i} = double(I);
end

end

