
import torch


DATA_PATH = 'data/data.csv'
FEATURE_NAMES = [
    'Temperature', 'Systolic Pressure', 'Diastolic Pressure',
    'Heart Rate', 'Respiratory Rate', 'BSL'
]
NUM_CLASSES = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


HIDDEN_SIZE = 140
NUM_LAYERS = 4
NUM_HEADS = 4
RANK = 32
DROPOUT = 0.2

BATCH_SIZE = 32
NUM_EPOCHS = 400
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0


LOSS_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESULTS_DIR = 'results'
LOSS_CURVE_PATH = f'{RESULTS_DIR}/loss_curve.png'
MICRO_ROC_PATH = f'{RESULTS_DIR}/micro_roc.png'
CLASS_ROC_PATH = f'{RESULTS_DIR}/class_roc.png'

CLASS_NAMES = ['Health', 'Mildness', 'Moderation', 'Severity', 'Extreme Severity']
