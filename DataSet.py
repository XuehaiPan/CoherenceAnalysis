import os
from typing import Dict, List, Tuple
import json
from Config import DATA_FILE_PATH, FIGURE_DIR, POSITIVE


def tokenize(sentence: str) -> List[str]:
    return list(filter(None, sentence.lower().split()))


def words_labels_generator(dataset: str) -> Tuple[List[str], int]:
    with open(file = DATA_FILE_PATH[dataset], encoding = 'UTF-8') as file:
        decoder: json.JSONDecoder = json.JSONDecoder()
        for line in file:
            data: Dict[str, str] = decoder.decode(s = line)
            yield tokenize(sentence = data['text']), int(data['label'])


def draw_data_distribution() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def get_seq_len(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        pos_seq_len: List[int] = []
        neg_seq_len: List[int] = []
        for words, label in words_labels_generator(dataset = dataset):
            if label == POSITIVE:
                pos_seq_len.append(len(words))
            else:
                neg_seq_len.append(len(words))
        return np.array(pos_seq_len, dtype = np.int32), np.array(neg_seq_len, dtype = np.int32)
    
    fig: plt.Figure
    axes: Dict[Tuple[int, int], plt.Axes]
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
    
    train_pos_seq_len: np.ndarray
    train_neg_seq_len: np.ndarray
    valid_pos_seq_len: np.ndarray
    valid_neg_seq_len: np.ndarray
    test_seq_len: np.ndarray
    train_pos_seq_len, train_neg_seq_len = get_seq_len(dataset = 'train')
    valid_pos_seq_len, valid_neg_seq_len = get_seq_len(dataset = 'valid')
    _, test_seq_len = get_seq_len(dataset = 'test')
    
    max_seq_len: int = max(train_pos_seq_len.max(), train_neg_seq_len.max(),
                           valid_pos_seq_len.max(), valid_neg_seq_len.max(),
                           test_seq_len.max())
    
    xticks: List[int] = list(range(0, max_seq_len + 100, 100)) + [320]
    try:
        xticks.remove(300)
    except ValueError:
        pass
    xticks.sort()
    
    sns.distplot(np.concatenate([train_pos_seq_len, train_neg_seq_len]), kde = False, label = 'Total', ax = axes[0, 0])
    sns.distplot(train_pos_seq_len, kde = False, label = 'Positive', ax = axes[0, 0])
    sns.distplot(train_neg_seq_len, kde = False, label = 'Negative', ax = axes[0, 0])
    axes[0, 0].axvline(x = 320, linestyle = '-.', color = 'black', alpha = 0.5)
    axes[0, 0].set_xticks(ticks = xticks)
    axes[0, 0].set_xlim(left = 0)
    axes[0, 0].set_xlabel('length')
    axes[0, 0].set_ylabel('frequency')
    axes[0, 0].set_title('Sequence Length in Training Data')
    axes[0, 0].legend()
    
    sns.distplot(np.concatenate([valid_pos_seq_len, valid_neg_seq_len]), kde = False, label = 'Total', ax = axes[0, 1])
    sns.distplot(valid_pos_seq_len, kde = False, label = 'Positive', ax = axes[0, 1])
    sns.distplot(valid_neg_seq_len, kde = False, label = 'Negative', ax = axes[0, 1])
    axes[0, 1].axvline(x = 320, linestyle = '-.', color = 'black', alpha = 0.5)
    axes[0, 1].set_xticks(ticks = xticks)
    axes[0, 1].set_xlim(left = 0)
    axes[0, 1].set_xlabel('length')
    axes[0, 1].set_ylabel('frequency')
    axes[0, 1].set_title('Sequence Length in Validation Data')
    axes[0, 1].legend()
    
    sns.distplot(test_seq_len, kde = False, ax = axes[1, 0])
    axes[1, 0].axvline(x = 320, linestyle = '-.', color = 'black', alpha = 0.5)
    axes[1, 0].set_xticks(ticks = xticks)
    axes[1, 0].set_xlim(left = 0)
    axes[1, 0].set_xlabel('length')
    axes[1, 0].set_ylabel('frequency')
    axes[1, 0].set_title('Sequence Length in Test Data')
    
    sns.distplot(np.concatenate([train_pos_seq_len, train_neg_seq_len,
                                 valid_pos_seq_len, valid_neg_seq_len,
                                 test_seq_len]),
                 kde = False, ax = axes[1, 1])
    axes[1, 1].axvline(x = 320, linestyle = '-.', color = 'black', alpha = 0.5)
    axes[1, 1].set_xticks(ticks = xticks)
    axes[1, 1].set_xlim(left = 0)
    axes[1, 1].set_xlabel('length')
    axes[1, 1].set_ylabel('frequency')
    axes[1, 1].set_title('Sequence Length in All Data')
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'data_dist.png'))
    fig.show()


def main() -> None:
    draw_data_distribution()


if __name__ == '__main__':
    main()
