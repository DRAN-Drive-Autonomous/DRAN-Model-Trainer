import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
import datetime
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# CONSTANTS
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
start_epochs = 0
RandomNormal = tf.keras.initializers.RandomNormal(stddev=0.02)

# Add Position Embeddings
class AddPositionEmbs(layers.Layer):
  """inputs are image patches
  Custom layer to add positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'posemb_init': self.posemb_init,
      })
      return config

def createCallbacks():
    timenow = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    log_dir = os.path.join(os.path.join("./train/", timenow), "logs/fit/")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = os.path.join(os.path.join("./train/", timenow), "checkpoint/fit/")
    if not os.path.exists(checkpoint_filepath):
        os.makedirs(checkpoint_filepath)
    checkpoint_filepath = os.path.join(checkpoint_filepath, "/weights.{epoch:02d}-{val_loss:.2f}.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )
    return [[tensorboard_callback, checkpoint_callback], [log_dir, checkpoint_filepath]]


def main():
    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Allow GPU memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU available. Training on GPU.")
    else:
        print("GPU not available. Training on CPU.")

    # Load data from NPZ file
    print("\nLoading and Preprocessing the data")
    X_train = np.load("./data_10000.npz", allow_pickle=True)
    y_train = X_train['y']
    X_train = X_train['X'] / 255.0
    # X = X['X']
    
    print("X shape: ", X_train.shape)
    print("X image shape: ", X_train[0][0].shape)
    print("X radar shape: ", X_train[0][1].shape)
    print("y shape: ", y_train.shape)

    # Load your model
    print("\nLoading the model")
    model_path = "./dranNN.h5"
    model = load_model(model_path, custom_objects={'AddPositionEmbs': AddPositionEmbs, 'RandomNormal': RandomNormal})

    # Compile your model
    print("\nCompiling model")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    loss_function = tf.keras.losses.BinaryCrossentropy()

    # Common classification metrics
    metrics = ['accuracy',
        tf.keras.metrics.MeanSquaredError(name='mse')]

    model.compile(
        optimizer=optimizer,
        loss=[loss_function] * 4,
        metrics=metrics,
    )   

    print("\nCreating callbacks")
    callbacks = createCallbacks()
    print(f"Checkpoint Filepath: {callbacks[1][1]}\nTensorboard Filepath: {callbacks[1][0]}")

    # Split data into training and validation sets
    print("\nSpliting dataset")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the model using GPU if available
    print("\nTraining model")
    # return
    with tf.device('/GPU:0'):
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(X_val, y_val),
            initial_epoch=start_epochs,
            callbacks=callbacks[0])
        
        model.load_weights(callbacks[1][1])

    # Save the trained model
    print("Saving model")
    return
    model.save("./dranNN.h5")

    return

if __name__ == "__main__":
    main()