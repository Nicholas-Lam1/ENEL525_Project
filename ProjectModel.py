import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, PReLU, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.inf)

#Define a function to compute the confusion matrix (you can also use "sklearn.metrics" library for a ready one)
def compute_confusion_matrix(true, pred):
    K = len(np.unique(true)) 
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    return result

def duplicate_upside_down(image, label):
    flipped_image = tf.image.flip_up_down(image)  
    return tf.concat([image[None], flipped_image[None]], axis=0), tf.concat([label[None], label[None]], axis=0)

dataset_dir = ".\\BMPDataset"
image_size = (256, 256)
batch_size = 100  # Number of images per batch
validation_split = 0.2  # Fraction of data to reserve for validation/testing

# ds_train = tf.keras.utils.image_dataset_from_directory(
#     dataset_dir,
#     labels = 'inferred',
#     label_mode = 'int',  
#     image_size = image_size,
#     batch_size = batch_size,
#     shuffle = True,
#     seed = 123, 
#     validation_split = validation_split,  
#     subset = "training",  
# )

# ds_val = tf.keras.utils.image_dataset_from_directory(
#     dataset_dir,
#     labels = 'inferred',  
#     label_mode = 'int',  
#     image_size = image_size,
#     batch_size = batch_size,
#     shuffle = True,  
#     seed = 123,  
#     validation_split = validation_split,  
#     subset = "validation",  
# )

# train_labels = np.concatenate([label.numpy() for _, label in ds_train], axis=0)
# print("Training set class distribution:", np.bincount(train_labels))
# val_labels = np.concatenate([label.numpy() for _, label in ds_val], axis=0)
# print("Training set class distribution:", np.bincount(val_labels))

# val_batches = tf.data.experimental.cardinality(ds_val)
# ds_val = ds_val.take((2*val_batches) // 3)
# ds_test = ds_val.skip((2*val_batches) // 3)

ds_full = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels="inferred",
    label_mode="int",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True, 
    seed=1,
)

# Get the total number of samples
full_batches = tf.data.experimental.cardinality(ds_full)
ds_train = ds_full.take((8 * full_batches) // 10)
ds_val = ds_full.skip((8 * full_batches) // 10)

train_labels = np.concatenate([label.numpy() for _, label in ds_train], axis=0)
print("Training set class distribution:", np.bincount(train_labels))
val_labels = np.concatenate([label.numpy() for _, label in ds_val], axis=0)
print("Training set class distribution:", np.bincount(val_labels))

val_batches = tf.data.experimental.cardinality(ds_val)
ds_val = ds_val.take((2 * val_batches) // 3)
ds_test = ds_val.skip((2 * val_batches) // 3)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label
 
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.cache()
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

ds_test = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_val.cache()
ds_test = ds_val.prefetch(tf.data.AUTOTUNE)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
    
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(96, (3, 3), activation = 'relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Dropout(0.1))  
model.add(MaxPooling2D(pool_size = (2, 2)))
 
model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))  
model.add(Dense(64, activation = 'relu')) 
model.add(Dropout(0.1))  

model.add(Dense(21, activation = 'softmax'))  

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.0005),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
)

early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True
)

history = model.fit(
    ds_train,
    epochs = 25,
    validation_data = ds_val,
    callbacks = [early_stopping],
)

model.summary()
model.save('my_model.keras')

test_labels = np.concatenate([label.numpy() for _, label in ds_test], axis=0)
results = model.predict(ds_test)
test_results = np.argmax(results, axis=1)

# print("Testing Prediction")
# print(test_labels)
# print(test_results)

confusion_matrix = compute_confusion_matrix(test_labels, test_results)

# Create a DataFrame for better readability
confusion_df = pd.DataFrame(confusion_matrix, columns=[f'Predicted {i}' for i in range(confusion_matrix.shape[1])], index=[f'True {i}' for i in range(confusion_matrix.shape[0])])

print("Confusion Matrix:")
print(confusion_df)
confusion_df.to_csv('confusion_matrix.csv', index=True)

plt.figure(figsize = (12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['sparse_categorical_accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], label = 'Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
