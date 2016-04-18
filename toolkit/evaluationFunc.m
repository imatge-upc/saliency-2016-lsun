function [ meanMetric, allMetrics] = evaluationFunc( result, data, metricName )
%EVALUATIONFUNC Evaluate result with the metric
%   result: array of cells containing the predicted saliency map
%   data: the ground truth data
%   metricName: the name of metric
%       -"similarity": Similarity
%       -"CC": CC
%       -"AUC_Borji": AUC_Borji
%       -"AUC_Judd": AUC_Judd
%       -"AUC_shuffled": sAUC
%   if the ground truth cannot be found, e.g. testing data, the central
%   gaussian will be taken as ground truth automatically.

GlobalParameters;
addpath(METRIC_DIR);
assert(length(result)==length(data));
availableMetric = {'similarity','CC', 'AUC_Judd', 'AUC_Borji', 'AUC_shuffled'};
assert(any(strcmp(metricName, availableMetric)));
if strcmp(metricName, 'AUC_shuffled')
    try
        load(TRAIN_DATA_PATH);
    catch
        fprintf('Training data missing!\n');
    end
end


fh = str2func(metricName);

allMetrics = zeros(length(data),1);
for i = 1:length(data)
    saliency_path = sprintf( SALIENCY_PATTERN, data(i).image);
    fixation_path = sprintf( FIXATION_PATTERN, data(i).image);
    
    if any(strcmp(metricName, {'similarity','CC'}))
        if exist(saliency_path, 'file')
            load(saliency_path);
            I = double(I);
            allMetrics(i) = fh( result{i}, I);
        else       
            allMetrics(i) = nan;
        end
    elseif any(strcmp(metricName, {'AUC_Judd', 'AUC_Borji'}))
        if exist(fixation_path, 'file')
            load(fixation_path);
            I = double(I);
            allMetrics(i) = fh( result{i}, I);
        else       
            allMetrics(i) = nan;
        end       
    elseif strcmp(metricName, 'AUC_shuffled')
        if exist(fixation_path, 'file')
            load(fixation_path);
            I = I>0;
            I = double(I);
            ids = randsample(length(training), 10);
            fixation_point = zeros(0,2);
            for k = 1:10
                rescale = size(result{i})./training(ids(k)).resolution;
                %pts = vertcat(training(ids(k)).gaze.fixation); isun
                pts = vertcat(training(ids(k)).gaze.fixations); %salicon
                fixation_point = [fixation_point; pts.*repmat(rescale, size(pts,1), 1)];
            end
            otherMap = makeFixationMap(size(result{i}), fixation_point);
            allMetrics(i) = fh( result{i}, I, otherMap);
        else       
            allMetrics(i) = nan;
        end 
    else
        allMetrics(i) = nan;
    end
end

meanMetric = mean(allMetrics);
end

