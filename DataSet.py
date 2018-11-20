import os
import json


data_dir = '/Users/PanXuehai/Desktop/Computer Science/Web Data Mining/Coherence Analysis/'
data_file_path = {
    name: os.path.join(data_dir, '{}_data'.format(name))
    for name in ('train', 'valid', 'test')
}

POSITIVE = 1
NEGATIVE = 0
UNDEFINED = -1


def tokenize(sentence):
    return list(filter(None, sentence.lower().split()))


def words_labels_generator(dataset):
    with open(file = data_file_path[dataset], encoding = 'UTF-8') as file:
        decoder = json.JSONDecoder()
        for line in file:
            data = decoder.decode(s = line)
            yield tokenize(sentence = data['text']), int(data['label'])


def main():
    def get_seq_len(dataset):
        pos_seq_len = []
        neg_seq_len = []
        for words, label in words_labels_generator(dataset = dataset):
            if label == POSITIVE:
                pos_seq_len.append(len(words))
            else:
                neg_seq_len.append(len(words))
        return np.array(pos_seq_len, dtype = np.int32), np.array(neg_seq_len, dtype = np.int32)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 12))
    train_pos_seq_len, train_neg_seq_len = get_seq_len(dataset = 'train')
    valid_pos_seq_len, valid_neg_seq_len = get_seq_len(dataset = 'valid')
    _, test_seq_len = get_seq_len(dataset = 'test')
    
    max_seq_len = max(train_pos_seq_len.max(), train_neg_seq_len.max(),
                      valid_pos_seq_len.max(), valid_neg_seq_len.max(),
                      test_seq_len.max())
    
    xticks = list(range(0, max_seq_len + 100, 100)) + [320]
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
    fig.savefig(fname = './data_dist.png')
    fig.show()


if __name__ == '__main__':
    main()
