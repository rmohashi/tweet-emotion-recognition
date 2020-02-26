from .lstm_conv_model import lstm_conv_model as lstm_conv
from .lstm_model import lstm_model as lstm
from .cnn_model import cnn_model as cnn

NLP_MODEL = {
  'lstm_conv': lstm_conv,
  'lstm': lstm,
  'cnn': cnn,
}
