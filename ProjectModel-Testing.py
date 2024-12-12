import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, PReLU # type: ignore
import numpy as np
import matplotlib.pyplot as plt

#Define a function to compute the confusion matrix (you can also use "sklearn.metrics" library for a ready one)
def compute_confusion_matrix(true, pred):
    K = len(np.unique(true)) #Number of classes
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

ds_train = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels = 'inferred',  # Automatically infer labels from subdirectory names
    label_mode = 'int',  # Labels are integers
    image_size = image_size,
    batch_size = batch_size,
    shuffle = True,
    seed = 123,  # Ensure reproducibility
    validation_split = validation_split,  # Specify the validation split
    subset = "training",  # This loads the training subset
)

ds_test = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels = 'inferred',  # Automatically infer labels from subdirectory names
    label_mode = 'int',  # Labels are integers
    image_size = image_size,
    batch_size = batch_size,
    shuffle = False,  # No need to shuffle the test set
    seed = 123,  # Ensure reproducibility
    validation_split = validation_split,  # Specify the validation split
    subset = "validation",  # This loads the validation/testing subset
)

# ds_train = ds_train.flat_map(lambda images, labels: tf.data.Dataset.from_tensors(duplicate_upside_down(images, labels))).unbatch()

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = Sequential()

model.add(Conv2D(32, (7, 7), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
    
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(96, (3, 3), activation = 'relu'))
model.add(Dropout(0.05))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(Dropout(0.05))  
model.add(MaxPooling2D(pool_size = (2, 2)))
 
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.05))  
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.05))  
model.add(Dense(21, activation = 'softmax'))  

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.0005),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()],
)

history = model.fit(
    ds_train,
    epochs = 25,
    validation_data = ds_test,
)

model.summary()

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
