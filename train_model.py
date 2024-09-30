import matplotlib.pyplot as plt
import tensorflow as tf

VOCAB_SIZE = 10000
MAX_LENGTH = 32
EMBEDDING_DIM = 32
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32
EPOCHS = 20

print("Load datasets")
train_data = tf.data.Dataset.load("train_data")
valid_data = tf.data.Dataset.load("valid_data")
test_data = tf.data.Dataset.load("test_data")

# Optimize the datasets for training
train_dataset_final = (train_data
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       )

valid_dataset_final = (valid_data
                       .cache()
                       .shuffle(SHUFFLE_BUFFER_SIZE)
                       .prefetch(PREFETCH_BUFFER_SIZE)
                       .batch(BATCH_SIZE)
                       )

test_dataset_final = (test_data
                      .cache()
                      .prefetch(PREFETCH_BUFFER_SIZE)
                      .batch(BATCH_SIZE)
                      )


def create_model(embed, lr):
    """Creates and compiles the model

    Returns:
        tf.keras.Model: compiled model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(MAX_LENGTH,)),
            tf.keras.layers.Embedding(VOCAB_SIZE, embed),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
       ]
                                )
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model

model = create_model(12, 1e-2)
history = model.fit(train_dataset_final, epochs=EPOCHS, validation_data=valid_dataset_final, verbose=2)

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.grid()
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

#evaluate model
results = model.evaluate(test_dataset_final)
print("test loss, test acc:", results)