from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential

def lstm_model(input_length,
               input_dim,
               embedding_dim=100,
               dropout=0.1,
               lstm_units=128,
               lstm_dropout=0.1,
               recurrent_dropout=0.1):
  model = Sequential()
  model.add(Embedding(
      input_dim=input_dim,
      output_dim=embedding_dim,
      input_shape=(input_length,)
  ))

  model.add(Bidirectional(
    GRU(lstm_units, return_sequences=True,
         dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
  ))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(2, activation='softmax'))
  return model