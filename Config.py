import os
import re
from typing import Dict, Pattern


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

POSITIVE: int = 1
NEGATIVE: int = 0
UNDEFINED: int = -1

BATCH_SIZE: int = 64
VEC_SIZE: int = 128
SEQ_LEN: int = 320

DATA_DIR: str = './data/'
DATA_FILE_PATH: Dict[str, str] = {
    name: os.path.join(DATA_DIR, f'{name}_data')
    for name in ('train', 'valid', 'test')
}

FIGURE_DIR: str = './figures/'
MODEL_DIR: str = './models/'
LOG_DIR: str = './logs/'

for DIR in (FIGURE_DIR, MODEL_DIR, LOG_DIR):
    if not os.path.exists(DIR):
        os.mkdir(DIR)

WORD2VEC_MODEL_PATH: str = os.path.join(MODEL_DIR, 'word2vec.model')
LATEST_MODEL_PATH: str = os.path.join(MODEL_DIR, 'latest.h5')
LOG_FILE_PATH: str = os.path.join(LOG_DIR, 'log.csv')

MODEL_FILE_PATTERN: Pattern = re.compile(r'.*epoch(?P<epoch>\d*)_acc(?P<val_acc>[\d.]*)\.h5')
MODEL_FMT_STR: str = os.path.join(MODEL_DIR, 'epoch{epoch:02d}_acc{val_acc:.4f}.h5')
