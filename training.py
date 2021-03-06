from typing import List, Tuple
import numpy as np
from tensorflow import keras
from dataset import words_labels_generator
from config import BATCH_SIZE, VEC_SIZE, SEQ_LEN, \
    MODEL_FMT_STR, MODEL_FILE_PATTERN, LATEST_MODEL_PATH, \
    LOG_DIR, LOG_FILE_PATH
from word2vec import get_word2vec_model
from model import get_model_paths, build_network


def vector_label_generator(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    word2vec = get_word2vec_model()
    for words, label in words_labels_generator(dataset = dataset):
        vectors: np.ndarray = np.zeros(shape = (SEQ_LEN, VEC_SIZE), dtype = np.float32)
        for i, word in zip(range(SEQ_LEN), words):
            vectors[i] = word2vec.get_vector(word)
        yield vectors, np.array([label], dtype = np.int32)


def get_all_vectors_labels(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    vectors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for vector, label in vector_label_generator(dataset = dataset):
        vectors.append(vector)
        labels.append(label)
    vectors: np.ndarray = np.array(vectors, dtype = np.float32)
    labels: np.ndarray = np.array(labels, dtype = np.int32)
    return vectors, labels


def train(epochs: int) -> None:
    train_vectors, train_labels = get_all_vectors_labels(dataset = 'train')
    valid_vectors, valid_labels = get_all_vectors_labels(dataset = 'valid')
    
    tensorBoard = keras.callbacks.TensorBoard(log_dir = LOG_DIR,
                                              histogram_freq = 0,
                                              batch_size = BATCH_SIZE,
                                              write_graph = True,
                                              write_grads = True,
                                              write_images = True)
    csvLogger = keras.callbacks.CSVLogger(filename = LOG_FILE_PATH,
                                          append = True)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = MODEL_FMT_STR,
                                                 monitor = 'val_acc',
                                                 verbose = 1)
    checkpointLatest = keras.callbacks.ModelCheckpoint(filepath = LATEST_MODEL_PATH,
                                                       monitor = 'val_acc',
                                                       verbose = 1)
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 5,
                                                  verbose = 1)
    
    try:
        model_paths = get_model_paths(sortby = 'epoch', reverse = False)
        initial_model_path = model_paths[-1]
        initial_epoch = int(MODEL_FILE_PATTERN.match(string = initial_model_path).group('epoch'))
    except IndexError:
        initial_epoch = 0
        initial_model_path = LATEST_MODEL_PATH
    
    model = build_network(model_path = initial_model_path)
    try:
        print(f'initial_epoch = {initial_epoch}')
        print(f'initial_model_path = {initial_model_path}')
        model.fit(x = train_vectors, y = train_labels,
                  batch_size = BATCH_SIZE, epochs = epochs, initial_epoch = initial_epoch,
                  validation_data = (valid_vectors, valid_labels), shuffle = True,
                  callbacks = [tensorBoard, csvLogger, checkpoint, checkpointLatest, earlyStopping],
                  workers = 4, use_multiprocessing = True)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(filepath = LATEST_MODEL_PATH)


def main() -> None:
    train(epochs = 40)


if __name__ == '__main__':
    main()
