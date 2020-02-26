from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential

def lstm_model(input_length,
               input_dim,
               num_classes,
               embedding_layer,
               embedding_dim=100,
               dropout=0.1,
               gru_units=128,
               gru_dropout=0.1,
               recurrent_dropout=0.1):
  model = Sequential()

  if embedding_layer:
    model.add(embedding_layer)
  else:
    model.add(Embedding(
        input_dim=input_dim,
        output_dim=embedding_dim,
        input_shape=(input_length,)
    ))

  model.add(Bidirectional(
    LSTM(gru_units, return_sequences=True,
         dropout=gru_dropout, recurrent_dropout=recurrent_dropout)
  ))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(dropout))

  model.add(Dense(num_classes, activation='sigmoid'))
  return model
