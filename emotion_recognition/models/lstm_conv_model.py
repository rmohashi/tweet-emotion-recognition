from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model

def lstm_conv_model(input_length,
                    input_dim,
                    num_classes,
                    embedding_layer,
                    embedding_dim=100,
                    spatial_dropout=0.2,
                    lstm_units=128,
                    lstm_dropout=0.1,
                    recurrent_dropout=0.1,
                    filters=64,
                    kernel_size=3):
  input_layer = Input(shape=(input_length,))

  if embedding_layer:
    output_layer = embedding_layer(input_layer)
  else:
    output_layer = Embedding(
      input_dim=input_dim,
      output_dim=embedding_dim,
      input_shape=(input_length,)
    )(input_layer)

  output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

  output_layer = Bidirectional(
    LSTM(lstm_units, return_sequences=True,
         dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
  )(output_layer)
  output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                        kernel_initializer='glorot_uniform')(output_layer)

  avg_pool = GlobalAveragePooling1D()(output_layer)
  max_pool = GlobalMaxPooling1D()(output_layer)
  output_layer = concatenate([avg_pool, max_pool])

  output_layer = Dense(num_classes, activation='softmax')(output_layer)

  model = Model(input_layer, output_layer)
  return model
