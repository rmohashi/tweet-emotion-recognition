from .lstm_conv_model import lstm_conv_model as lstm_conv
from .gru_model import gru_model as gru

NLP_MODEL = {
  'gru': gru,
  'lstm_conv': lstm_conv,
}