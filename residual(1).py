import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Dropout

#tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img

#essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
#from matplotlib import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint

#tensorflow, keras
import keras
from keras.layers import merge, Conv2D, Input, Reshape, Activation, Add, Multiply
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
from numpy import genfromtxt
from keras.preprocessing.image import img_to_array as img_to_array
from keras.preprocessing.image import load_img as load_img

#essentials
import numpy as np
import math
import os
import random
import pandas as pd
import tensorflow as tf
import argparse
#from matplotlib.pyplot import pyplot as plt
from keras import callbacks
from keras.callbacks import ModelCheckpoint

#attention needed
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Reshape
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random

def hack_model(input_shape):
    
    X_input = Input(input_shape)

    # First Block
    #CONV layer
    conv_1_in = Conv2D(128, (7, 7), strides = (1,1), activation='relu', padding='same')(X_input)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2,2), strides = 1, padding='same')(conv_1_in)
    conv_1_out = BatchNormalization(axis=-1)(X)

    conv_1_in_2 = Conv2D(128, (5,5), strides = (1,1), activation='relu', padding='same')(conv_1_in)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2,2), strides = 1, padding='same')(conv_1_out)
    conv_1_out_2 = BatchNormalization(axis=-1)(X)
    
    conv1= Add()([conv_1_out, conv_1_out_2])
    
    # First Block
    #CONV layer
    X = Conv2D(128, (5, 5), strides = (1,1), activation='relu', padding='same')(conv1)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2,2), strides = 1, padding='same')(X)
    conv_2_out = BatchNormalization(axis=-1)(X)

    
    #skip conn
    X = Conv2D(128, (5,5), strides = (1,1), activation='relu', padding='same')(conv1)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2,2), strides = 1, padding='same')(X)
    skip_out = BatchNormalization(axis=-1)(X)

    skip_out= Add()([skip_out, conv_2_out])
    # First Block
    #CONV layer
    conv_3_in = Conv2D(128, (5, 5), strides = (1,1), activation='relu', padding='same')(skip_out)
    X = Conv2D(64, (7, 7), strides = (1,1), activation='relu', padding='same')(X)
    # MAXPOOL + BatchNorm
    X = MaxPooling2D((2,2), strides = 1, padding='same')(X)
    conv_3_out = BatchNormalization(axis=-1)(X)

    #next block
    X = Conv2D(64, (5, 5), strides = (1,1), activation='relu', padding='same')(conv_3_out)
    # MAXPOOL
    X = MaxPooling2D((2,2), strides = 1, padding='same')(X)
    X = BatchNormalization(axis=-1)(X)

    X= Add()([X, conv_3_out])
    
    conv_3_out = Conv2D(32, (5,5), strides = (1, 1),activation='relu', padding='same')(X)
    # MAXPOOL
    X = MaxPooling2D(pool_size = 3, strides = 1, padding='same')(conv_3_out)
    X = BatchNormalization(axis=-1)(X)

    # Top layer
    #X = AveragePooling2D(pool_size=(3,3), strides=(1,1))(X)
    X = Conv2D(16, (3,3), strides = (1,1),activation='relu', padding='same')(X)

    # L2 normalization
    X = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(X)

    out = Conv2D(5, (1, 1), strides = (1,1),activation='softmax')(X)

    model = Model(inputs = X_input, outputs = out)
    
    return model

X_train = np.load('X_only.npy')
y_train_damage_grade = np.load('y_only.npy')

y_tr = y_train_damage_grade[:393216].reshape(6,256,256, 5)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# pca.fit(X_train)

# X_train_pca = pca.transform(X_train)

X_train = X_train[:393216].reshape(6,256,256, 9)

#print(X_tr.shape)

with tf.device('/device:GPU:0'):

    tf.reset_default_graph()

    model= hack_model((256,256,9))

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    
    model.compile(optimizer = opt , loss = 'categorical_crossentropy', metrics = ["accuracy"]);
    
    ## Printing the modle summary
    print(model.summary())
    try:
        model= load_model('modelResidual.h5')
        history = model.fit(X_train, y_tr, epochs= 500, batch_size=6)
        print(history.history.keys())
        print(history.history['acc'])
        print(history.history['loss'])
    except:
        model.save('modelResidual2.h5')
        model.save('modelResidual2.h5')