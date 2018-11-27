import os
from glob import glob
from tensorflow import keras
from Config import VEC_SIZE, SEQ_LEN, MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN


def get_model_paths(sortby = 'epoch', reverse = False):
    assert sortby in ('epoch', 'acc', 'val_acc')
    if 'acc' in sortby:
        sortby = 'val_acc'
    model_paths = glob(pathname = os.path.join(MODEL_DIR, 'epoch*_acc*.h5'))
    model_paths.sort(key = lambda file: float(MODEL_FILE_PATTERN.match(file).group(sortby)), reverse = reverse)
    return model_paths


def build_network(model_path = None):
    if model_path is None:
        try:
            model_paths = get_model_paths(sortby = 'val_acc', reverse = True)
            model_path = model_paths[0]
            print('best_model_path = {}'.format(model_path))
        except IndexError:
            model_path = LATEST_MODEL_PATH
    
    try:
        return keras.models.load_model(filepath = model_path)
    except OSError:
        pass
    
    # Inputs
    inputs = keras.layers.Input(shape = (SEQ_LEN, VEC_SIZE), name = 'inputs')
    
    # LSTM block 1
    LSTM_1 = keras.layers.LSTM(units = 128, input_shape = (SEQ_LEN, VEC_SIZE),
                               dropout = 0.2, return_sequences = True, name = 'LSTM_1')
    
    x = LSTM_1(inputs)
    
    # LSTM block 2
    LSTM_2 = keras.layers.LSTM(units = 128, input_shape = (SEQ_LEN, 128),
                               dropout = 0.2, return_sequences = True, name = 'LSTM_2')
    
    x = LSTM_2(x)
    
    # LSTM block 3
    LSTM_3 = keras.layers.LSTM(units = 128, input_shape = (SEQ_LEN, 128),
                               dropout = 0.2, return_state = True, name = 'LSTM_3')
    
    _, _, x = LSTM_3(x)
    
    # Dense block 4
    Dense_4 = keras.layers.Dense(units = 64, input_dim = 128,
                                 activation = None, name = 'Dense_4')
    BatchNorm_4 = keras.layers.BatchNormalization(name = 'BatchNorm_4')
    Tanh_4 = keras.layers.Activation(activation = 'tanh', name = 'Tanh_4')
    
    x = Dense_4(x)
    x = BatchNorm_4(x)
    x = Tanh_4(x)
    
    # Dense block 5
    Dense_5 = keras.layers.Dense(units = 32, input_dim = 64,
                                 activation = None, name = 'Dense_5')
    BatchNorm_5 = keras.layers.BatchNormalization(name = 'BatchNorm_5')
    Tanh_5 = keras.layers.Activation(activation = 'tanh', name = 'Tanh_5')
    
    x = Dense_5(x)
    x = BatchNorm_5(x)
    x = Tanh_5(x)
    
    # Dense block 6
    Dense_6 = keras.layers.Dense(units = 1, input_dim = 32,
                                 activation = None, name = 'Dense_6')
    Sigmoid_6 = keras.layers.Activation(activation = 'sigmoid', name = 'Sigmoid_6')
    
    x = Dense_6(x)
    outputs = Sigmoid_6(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    RMSprop_Optimizer = keras.optimizers.RMSprop(lr = 0.001, decay = 1E-4)
    model.compile(optimizer = RMSprop_Optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    model.save(filepath = LATEST_MODEL_PATH)
    
    return model
