from .lstm_conv_model import lstm_conv_model as lstm_conv
from .lstm_model import lstm_model as lstm
from .cnn_model import cnn_model as cnn
from .fasttext_model import fasttext_model as fasttext

NLP_MODEL = {
  'lstm': lstm,
  'lstm_conv': lstm_conv,
  'cnn': cnn,
  'fasttext': fasttext,
}
