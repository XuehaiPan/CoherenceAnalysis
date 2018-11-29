import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from config import BATCH_SIZE, LOG_FILE_PATH, FIGURE_DIR
from model import get_model_paths, build_network
from training import get_all_vectors_labels


def draw_train_logs() -> None:
    logs: pd.DataFrame = pd.read_csv(LOG_FILE_PATH)
    logs['epoch'] += 1
    
    print(logs)
    
    fig: plt.Figure
    axes: List[plt.Axes]
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    
    axes[0].plot(logs['epoch'], logs['acc'], label = 'train')
    axes[0].plot(logs['epoch'], logs['val_acc'], label = 'validation')
    axes[0].legend()
    axes[0].set_xticks(ticks = list(range(0, 41, 2)))
    axes[0].set_yticks(ticks = np.arange(start = 0.5, stop = 0.8, step = 0.025))
    axes[0].set_xlim(left = 0)
    axes[0].grid(axis = 'y')
    axes[0].set_xlabel(xlabel = 'epoch')
    axes[0].set_ylabel(ylabel = 'accuracy')
    axes[0].set_title(label = 'Accuracy')
    
    axes[1].plot(logs['epoch'], logs['loss'], label = 'train')
    axes[1].plot(logs['epoch'], logs['val_loss'], label = 'validation')
    axes[1].legend()
    axes[1].set_xticks(ticks = list(range(0, 41, 2)))
    axes[1].set_xlim(left = 0)
    axes[1].grid(axis = 'y')
    axes[1].set_xlabel(xlabel = 'epoch')
    axes[1].set_ylabel(ylabel = 'loss')
    axes[1].set_title(label = 'Loss')
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'training_log.png'))
    fig.show()


def predict(x: np.ndarray) -> np.ndarray:
    model_paths: List[str] = get_model_paths(sortby = 'val_acc', reverse = True)
    top5_model_path: List[str] = model_paths[:5]
    
    print(f'top5_model_path = {top5_model_path}')
    
    top5_predictions: List[np.ndarray] = []
    
    for no, model_path in zip(('1st', '2nd', '3rd', '4th', '5th'), top5_model_path):
        print(f'Use {no} best model {model_path}:')
        model: keras.Model = build_network(model_path = model_path)
        valid_predictions: np.ndarray = model.predict(x = x, batch_size = BATCH_SIZE, verbose = 1,
                                                      workers = 4, use_multiprocessing = True)
        valid_predictions = (valid_predictions >= 0.5).astype(dtype = np.int32)
        top5_predictions.append(valid_predictions)
    
    predictions: np.ndarray = np.mean(top5_predictions, axis = 0)
    predictions = (predictions >= 0.5).astype(dtype = np.int32)
    
    return predictions


def evaluate_on_valid_data() -> None:
    valid_vectors: np.ndarray
    valid_labels: np.ndarray
    valid_vectors, valid_labels = get_all_vectors_labels(dataset = 'valid')
    
    valid_predictions: np.ndarray = predict(x = valid_vectors)
    
    val_acc: float = (valid_labels == valid_predictions).astype(dtype = np.int32).mean()
    
    print(f'ensemble model\'s accuracy on validation data: {val_acc:.4f}')


def predict_on_test_data() -> None:
    test_vectors: np.ndarray
    test_vectors, _ = get_all_vectors_labels(dataset = 'test')
    
    test_labels: np.ndarray = predict(x = test_vectors)
    
    with open(file = './test_labels', mode = 'w', encoding = 'UTF-8') as out:
        for label in test_labels.flatten():
            out.write(f'{label}\n')


def main() -> None:
    draw_train_logs()
    
    evaluate_on_valid_data()
    predict_on_test_data()


if __name__ == '__main__':
    main()
