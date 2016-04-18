function [ result ] = algorithmFunc( img )
%ALGORITHMFUNC A naive central gaussian estimation
%   
GlobalParameters;

load(CENTRAL_PATH);
result = imresize(center, [size(img,1) size(img,2)], 'cubic');

end

