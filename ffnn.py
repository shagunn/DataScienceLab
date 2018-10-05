'''
Created on Apr 2, 2018

@author: dusans
'''

from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers

import sklearn.metrics
from sklearn.model_selection import GridSearchCV 

import numpy as np
from sklearn.metrics.scorer import make_scorer


class FFNN(object):
    
    _inputs = 0
    _layers = 0
    _units  = 0
    _model  = None
    _model_fn = ''
    
    
    def __init__(self, n_inputs):
        self._inputs = n_inputs
    
    
    def create(self, 
               n_layers,
               n_units, 
               activation_fcn='relu',
               lrate=1e-3,
               dropout_rate=0.,
               l2lambda=0.):
        '''
        Create a feed-forward neural network with given parameters.
        
        @param layer_units: list of values for number of neurons in each layer
        @param n_layers: number of layers in the network
        @param n_units: number of neurons per layer (same for all layers)
        @param activation_fn: activation function to use in each layer
        @param lrate: learning rate during training
        @param dropout_rate: dropout rate after each hidden layer    
        '''
        
        # input layer
        inputs = Input(shape=(self._inputs,))

        # create feed-forward layers
        layer_units = [n_units] * n_layers
        layers = FFNN.create_hidden_layers(layer_units, 
                                           inputs, 
                                           activation_fcn, 
                                           dropout_rate,
                                           l2lambda)
        
        # output layer
        output = Dense(1, activation='sigmoid')(layers)

#         self._model = Model(inputs, output)
        self._model = PrModel(inputs, output)
        
        # compile
        optimizer = Adam(lr=lrate)
        self._model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        
    def train(self, 
              x_train, y_train, 
              x_val, y_val, 
              epochs, 
              batch_size=128, 
              model_fn='trained_model.h5',
              verbose=0):
        '''
        Train instantiated neural network.
        
        @param x_train: training inputs
        @param y_train: training outputs
        @param x_val:   validation inputs (defaults to training inputs if None)
        @param y_val:   validation outputs (defaults to training outputs if None)
        @param epochs:  number of epochs for learning
        @param batch_size: batch size during training 
        @param model_fn:filename to save the best model during training
        @param verbose: display info during training   
        '''
        
        if model_fn is None or model_fn == '':
            clbk_lst = []
        else:
            clbk_save = ModelCheckpoint(model_fn,
                                        monitor='val_loss',
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=True,
                                        mode='auto')
            clbk_lst = [clbk_save]
        
        if x_val is None or y_val is None:
            x_val = x_train
            y_val = y_train
        
        self._model_fn = model_fn
        
        self._model.fit(x_train, 
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=clbk_lst, 
                        validation_data=(x_val, y_val),
                        verbose=verbose)
        
    
#     def predict_proba(self, x):
#         yh = self._model.predict(x)
#         return np.hstack((1 - y1, yh))
    def predict_proba(self, x):
        return self._model.predict_proba(x)
    
#     def predict(self, x, dtype=np.int32):
#         yh = self._model.predict(x) > 0.5
#         return yh.astype(dtype)

    def predict(self, x):
        return self._model.predict(x)
    
    def predict_classes(self, x):
        return self._model.predict_classes(x)
    
    
    def load_model(self, model_fn):
        if model_fn is None or model_fn == '':
            model_fn = self._model_fn
            
        if self._model is None:
            raise ValueError('model not created')
        
        self._model.load_weights(model_fn)
    
    
    @staticmethod
    def create_ffnn(n_inputs = 1,
                    n_layers = 1,
                    n_units = 128,
                    activation_fcn='relu',
                    lrate=1e-3,
                    dropout_rate=0.,
                    l2lambda=0.01):
        M = FFNN(n_inputs)
        M.create(n_layers, n_units, activation_fcn, lrate, dropout_rate, l2lambda)
        return M._model
        
    
    
    @staticmethod
    def create_hidden_layers(layer_units, 
                             inputs, 
                             activation_fcn='relu', 
                             dropout_rate=0.,
                             l2lambda=0.):
        '''
        Create at least one Dense layer 
        '''
        if not isinstance(layer_units, list):
            layer_units = [layer_units]
        
        for neur in layer_units:
            units = int(neur)
            layer = Dense(units, 
                          activation=activation_fcn, 
                          kernel_regularizer=regularizers.l2(l2lambda))(inputs)
            if dropout_rate > 0:
                layer = Dropout(dropout_rate)(layer)
            inputs = layer
            
        return layer
    


def validate_parameters(x, y, param_grid, folds, verbose):
    model = KerasClassifier(FFNN.create_ffnn, 
                            n_inputs=x.shape[1])

    scorer = make_scorer(sklearn.metrics.log_loss, 
                         greater_is_better=False)

    gcv = GridSearchCV(estimator=model, 
                       param_grid=param_grid, 
                       scoring=scorer,
                       n_jobs=1,
                       cv=folds,
                       refit=False,
                       verbose=verbose)
        
    gcv_res = gcv.fit(x, y)
       
    return gcv_res


class PrModel(Model):
    
    def predict_classes(self, x, **kwargs):
        yh = self.predict(x, **kwargs) > 0.5
        return yh.astype(np.int32)
        
    def predict_proba(self, x, **kwargs):
        yh = self.predict(x, **kwargs)
        return np.hstack((1 - yh, yh))
        
        
        

if __name__ == '__main__':
    # random testing stuff
    ni = 12
    N = 1000
    
    x = np.random.random((N, ni))
    y = np.random.randint(low=0, high=2, size=(N, 1))
    
    m = FFNN(ni)
    
    m.create(n_layers=3, 
             n_units=16,
             activation_fcn='selu', 
             lrate=1e-3, 
             dropout_rate=0.,
             l2lambda=0.01)
    m._model.summary()
        
    m.train(x[:900], y[:900], x[900:], y[900:], 10, 'rnddata.hdf5')
    yh = m.predict(x)


#     # validation phase    
#     param_grid = {'n_layers': [2, 3],
#                   'n_units': [16],
#                   'activation_fcn': ['relu', 'tanh'],
#                   'lrate': [1e-3],
#                   'dropout_rate': [0.],
#                   'l2lambda': [0.],
#                   'epochs': [10],
#                   'batch_size': [128]
#                   }
#     cv_res = validate_parameters(x, y, param_grid)
    
    