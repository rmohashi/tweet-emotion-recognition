from .lstm_conv_model import lstm_conv_model as lstm_conv
from .lstm_model import lstm_model as lstm

NLP_MODEL = {
  'lstm': lstm,
  'lstm_conv': lstm_conv,
}