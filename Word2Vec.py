from gensim.models import Word2Vec, KeyedVectors
from Config import VEC_SIZE, WORD2VEC_MODEL_PATH


def get_word2vec_model():
    try:
        return KeyedVectors.load(WORD2VEC_MODEL_PATH)
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
            print(f'Epoch #{self.epoch} start')
        
        def on_epoch_end(self, model):
            print(f'Epoch #{self.epoch} end')
            print(f'Save model to {WORD2VEC_MODEL_PATH}')
            model.wv.save(WORD2VEC_MODEL_PATH)
    
    sentences = []
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'train')))
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'valid')))
    sentences.extend(map(lambda tokensLabel: tokensLabel[0], words_labels_generator(dataset = 'test')))
    
    epoch_logger = EpochLogger()
    
    model = Word2Vec(sentences = sentences, size = VEC_SIZE, min_count = 1, workers = 4)
    model.train(sentences = sentences, total_examples = len(sentences), epochs = 40, callbacks = [epoch_logger])
    
    model.wv.save(WORD2VEC_MODEL_PATH)
    
    return KeyedVectors.load(WORD2VEC_MODEL_PATH)


def main():
    get_word2vec_model()


if __name__ == '__main__':
    main()
