from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D

def fasttext_model(input_length,
              input_dim,
              embedding_layer,
              embedding_dim=100):
  model = Sequential()

  model.add(Embedding(input_dim,
                      embedding_dim,
                      input_length=input_length))

  model.add(GlobalAveragePooling1D())

  model.add(Dense(1, activation='sigmoid'))

  return model
