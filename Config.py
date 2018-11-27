import os
import re


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

POSITIVE = 1
NEGATIVE = 0
UNDEFINED = -1

BATCH_SIZE = 64
VEC_SIZE = 128
SEQ_LEN = 320

DATA_DIR = './data/'
DATA_FILE_PATH = {
    name: os.path.join(DATA_DIR, f'{name}_data')
    for name in ('train', 'valid', 'test')
}

FIGURE_DIR = './figures/'
MODEL_DIR = './models/'
LOG_DIR = './logs/'

for DIR in (FIGURE_DIR, MODEL_DIR, LOG_DIR):
    if not os.path.exists(DIR):
        os.mkdir(DIR)

WORD2VEC_MODEL_PATH = os.path.join(MODEL_DIR, 'word2vec.model')
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, 'latest.h5')
LOG_FILE_PATH = os.path.join(LOG_DIR, 'log.csv')

MODEL_FILE_PATTERN = re.compile(r'.*epoch(?P<epoch>\d*)_acc(?P<val_acc>[\d.]*)\.h5')
MODEL_FMT_STR = os.path.join(MODEL_DIR, 'epoch{epoch:02d}_acc{val_acc:.4f}.h5')
