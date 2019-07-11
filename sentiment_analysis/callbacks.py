from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

def early_stopping(patience):
  return EarlyStopping(
    monitor='val_acc',
    mode='max',
    patience=patience
  )

def checkpoints(filepath, save_weights_only=False):
  return ModelCheckpoint(
    filepath=filepath,
    monitor='val_acc',
    save_best_only=True,
    mode='max',
    save_weights_only=save_weights_only,
    verbose=1
  )

def tensorboard(log_dir, batch_size):
  return TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    batch_size=batch_size
  )

def reduce_lr(patience):
  return ReduceLROnPlateau(
    monitor='val_acc',
    factor=0.5,
    patience=patience,
    min_lr=1e-6
  )