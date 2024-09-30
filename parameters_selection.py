import tensorflow as tf
import keras_tuner as kt

VOCAB_SIZE = 10000
MAX_LENGTH = 32
SHUFFLE_BUFFER_SIZE = 500
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

train_data = tf.data.Dataset.load("train_data")
valid_data = tf.data.Dataset.load("valid_data")
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

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(MAX_LENGTH,)))

  # Tune the embedding dimension
  # Choose an optimal value between 8-32
  hp_embed = hp.Int('embedding', min_value=8, max_value=32, step=2)
  model.add(tf.keras.layers.Embedding(VOCAB_SIZE, hp_embed))
  filters = hp.Int('filters', min_value=32, max_value=250, step=32)
  kernel_size = hp.Int('kernel_size', min_value=2, max_value=8, step=1)
  model.add(tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'))
  model.add(tf.keras.layers.GlobalMaxPooling1D())
  units = hp.Int('units', min_value=2, max_value=12, step=2)
  model.add(tf.keras.layers.Dense(units, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

  # Tune the learning rate for the optimizer
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
  optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
  model.compile(optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"])

  return model

tuner = kt.Hyperband(model_builder,
                     max_epochs=20,
                     objective="val_accuracy",
                     overwrite=True,
                     directory="tuner_dir",
                     project_name="sarcasm_detection"
                     )
tuner.search(train_dataset_final,
             epochs=20,
             validation_data = valid_dataset_final
             )

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete.\n
 The optimal embedding dimension is {best_hps.get('embedding')}. \n 
 The optimal learning rate is {best_hps.get('learning_rate')}. \n
 Number units {best_hps.get('units')} \n
 Kernel size {best_hps.get('kernel_size')} \n
 Filters number {best_hps.get('filters')} \n
""")
