%% Demo.m 
% All the codes in "code_forMetrics" are from MIT Saliency Benchmark (https://github.com/cvzoya/saliency). Please refer to their webpage for more details.

% load global parameters, you should set up the "ROOT_DIR" to your own path
% for data.
GlobalParameters;
addpath(genpath(METRIC_DIR));

% an example of running algorithm on validation set and evaluate with
% metrics.
load(VALIDATION_DATA_PATH);
results = predictFunc(validation(1:5000));
[meanMetric, allMetric] = evaluationFunc(results, validation(1:5000), 'similarity');
fprintf('Mean similarity: %3.2f\n', meanMetric);
[meanMetric, allMetric] = evaluationFunc(results, validation(1:5000), 'AUC_Judd');
fprintf('Mean AUC_Judd: %3.2f\n', meanMetric);
[meanMetric, allMetric] = evaluationFunc(results, validation(1:5000), 'AUC_shuffled');
fprintf('Mean AUC_shuffled: %3.2f\n', meanMetric);
[meanMetric, allMetric] = evaluationFunc(results, validation(1:5000), 'AUC_Borji');
fprintf('Mean AUC_Borji: %3.2f\n', meanMetric);
[meanMetric, allMetric] = evaluationFunc(results, validation(1:5000), 'CC');
fprintf('Mean CC: %3.2f\n', meanMetric);

% What to submit:
% participants are supposed to organize result in the format of "results".
% "results" is an array of cell, each of which contains the saliency map of
% each image.

%load(TEST_DATA_PATH);
%results = predictFunc(testing(1:10));
%[meanMetric, allMetric] = evaluationFunc(results, testing(1:10), 'similarity');
%fprintf('Mean similarity: %3.2f\n', meanMetric);
