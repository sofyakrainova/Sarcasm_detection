import matplotlib.pyplot as plt
import tensorflow as tf

VOCAB_SIZE = 10000
MAX_LENGTH = 32
EMBEDDING_DIM = 16
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32
FILTERS = 128
KERNEL_SIZE = 5
DENSE_DIM = 6
EPOCHS = 20

print("Load datasets")
train_data = tf.data.Dataset.load("train_data")
test_data = tf.data.Dataset.load("test_data")

# Optimize the datasets for training
train_dataset_final = (train_data
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
# Build the model

model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    tf.keras.layers.Conv1D(FILTERS, KERNEL_SIZE, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Print the model summary
model.summary()

# Compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# Train the model
history = model.fit(train_dataset_final, epochs=EPOCHS, validation_data=test_dataset_final, verbose=2)

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Plot the accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
