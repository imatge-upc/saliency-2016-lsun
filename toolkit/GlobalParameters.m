
ROOT_DIR = '/imatge/jpan/salicon_data/';

IMAGE_DIR = [ROOT_DIR 'images/'];
SALIENCY_DIR = [ROOT_DIR 'saliency/'];
FIXATION_DIR = [ROOT_DIR 'fixation/'];

VAL_TEST_DIR = ['/imatge/jpan/work/lsun2016/results/19-04-2016/'];

METRIC_DIR = 'code_forMetrics';

IMAGE_PATTERN = [IMAGE_DIR '%s' '.jpg'];
SALIENCY_PATTERN = [SALIENCY_DIR '%s' '.mat'];
FIXATION_PATTERN = [FIXATION_DIR '%s' '.mat'];
VAL_TEST_PATTERN = [VAL_TEST_DIR '%s' '.mat'];
CENTRAL_PATH = './center.mat';

TRAIN_DATA_PATH = [ROOT_DIR 'training.mat'];
VALIDATION_DATA_PATH = [ROOT_DIR 'validation.mat'];
TEST_DATA_PATH = [ROOT_DIR 'testing.mat'];
