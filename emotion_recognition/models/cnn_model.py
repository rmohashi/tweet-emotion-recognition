from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

def cnn_model(input_length,
              input_dim,
              num_classes,
              embedding_layer,
              embedding_dim=100,
              dropout=0.2,
              hidden_dim=250,
              filters=250,
              kernel_size=3):
  model = Sequential()

  if embedding_layer:
    model.add(embedding_layer)
  else:
    model.add(Embedding(input_dim,
                        embedding_dim,
                        input_length=input_length))

  model.add(Dropout(dropout))

  model.add(Conv1D(filters,
                   kernel_size,
                   padding='valid',
                   activation='relu',
                   strides=1))
  model.add(GlobalMaxPooling1D())

  model.add(Dense(hidden_dim, activation='relu'))

  model.add(Dense(num_classes, activation='sigmoid'))

  return model
