import os
from gensim.models import Word2Vec, KeyedVectors


VEC_SIZE = 128

MODEL_DIR = './models/'
LAST_MODEL_PATH = os.path.join(MODEL_DIR, 'word2vec.model')


def get_word2vec_model():
    try:
        return KeyedVectors.load(LAST_MODEL_PATH)
    except OSError:
        pass
    
    from DataSet import words_labels_generator
    from gensim.models.callbacks import CallbackAny2Vec
    
    class EpochLogger(CallbackAny2Vec):
        """Callback to log information about training"""
        
        def __init__(self):
            self.epoch = 0
        
        def on_train_begin(self, model):
            print('Start training Word2Vec model')
        
        def on_epoch_begin(self, model):
            self.epoch += 1
            print('Epoch #{} start'.format(self.epoch))
        
        def on_epoch_end(self, model):
            print('Epoch #{} end'.format(self.epoch))
            print('Save model to {}'.format(LAST_MODEL_PATH))
            model.wv.save(LAST_MODEL_PATH)
    
    sentences = []
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'train')))
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'valid')))
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'test')))
    
    epoch_logger = EpochLogger()
    
    model = Word2Vec(sentences = sentences, size = VEC_SIZE, min_count = 1, workers = 4)
    model.train(sentences = sentences, total_examples = len(sentences), epochs = 40, callbacks = [epoch_logger])
    
    model.wv.save(LAST_MODEL_PATH)
    
    return KeyedVectors.load(LAST_MODEL_PATH)


def main():
    get_word2vec_model()


if __name__ == '__main__':
    main()