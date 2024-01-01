
# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

from tensorflow import image
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.saving import save

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' ## Stops kernal error bug with matplotlib


# %%

work_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph_classifier_simple_data"

BATCH_SIZE : int = 32
IMG_HEIGHT : int = 32
IMG_WIDTH : int = 32
VAL_SPLIT : int = 0.2
SEED : int = 123
EPOCHS : int = 3

# %%

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    work_path,
    validation_split=VAL_SPLIT,
    labels='inferred',
    subset='both',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# %%

# Check to make sure data loaded correctly

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch)
    break

# %%

print(type(train_ds))
train_ds.c

# %%

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# output layer size
num_classes = len(class_names)

model = Sequential([
    # Standardize values to be in the [0, 1] range by using tf.keras.layers.Rescaling
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])


# %%

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# %%

model.summary()

# %%

# returns History object
fitted_model = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# %%

# History.history is a dict with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
print(fitted_model.history.keys())

# Note: metrics_names are available only after a keras.Model has been trained/evaluated on actual data.
print(model.metrics_names)

# %%

acc = fitted_model.history['accuracy']
val_acc = fitted_model.history['val_accuracy']

loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %%

validation_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset\\bar_resized"

test_dataset = tf.data.Dataset.list_files("C:/Users/pat_h/OneDrive/Desktop/Graph Classifier/graph dataset/bar_resized/*.jpg")

for i in test_dataset:
   print(tf.get_static_value(i))
   break

# %%

#### OKAY this looks promising

evaluate_data = []

for i in test_dataset:
    with Image.open(tf.get_static_value(i).decode('utf-8')) as im:
        im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)
    evaluate_data.append(im_array)



# %%

print(len(evaluate_data))

for i in range(10):
    print(evaluate_data[i].shape)


# %%

tensor_data = map(tf.convert_to_tensor, evaluate_data)
print(type(tensor_data))
tensor_data = list(tensor_data)
print(type(tensor_data))



# %%

######### IT FINALLY WORKED HOLY LORD DELIVER ME
tensor_data = tf.data.Dataset.from_tensors(tensor_data)


######### YEEEESSSSSS it's coming together
print(type(tensor_data))
print(tensor_data.get_single_element()[:10])


# %%

pred_new = model.predict(tensor_data, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

# Generates output predictions for the input samples.

# Computation is done in batches. This method is designed for batch processing of large numbers of inputs. 
# It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.

# %%

print(type(pred_new))
print(len(pred_new))
print(pred_new.shape)


# %%

pred_new = pd.DataFrame(pred_new, columns=['graph', 'natural'])
pred_new.head(n=20)


# %%

#### Test to see what happens when I use the natural images

validation_path_false = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset\\cifar"

test_dataset_false = tf.data.Dataset.list_files("C:/Users/pat_h/OneDrive/Desktop/Graph Classifier/graph dataset/cifar/*.jpg")


# %%

evaluate_data_false = []

for i in test_dataset_false:
    with Image.open(tf.get_static_value(i).decode('utf-8')) as im:
        im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)
    evaluate_data_false.append(im_array)


# %%

tensor_data_false = map(tf.convert_to_tensor, evaluate_data_false)
tensor_data_false = list(tensor_data_false)
tensor_data_false = tf.data.Dataset.from_tensors(tensor_data_false)
print(type(tensor_data_false))


# %%

pred_new_false = model.predict(tensor_data_false, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)


# %%

print(type(pred_new_false))
print(len(pred_new_false))
print(pred_new_false.shape)

# %%

pred_new_false = pd.DataFrame(pred_new_false, columns=['graph', 'natural'])
pred_new_false.head(n=20)

# %%

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# %%

print(type(model))


# %%

save_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\saved_models\\simple_classifier"

model.save((os.path.join(save_path, 'simple_classifier_1.keras')), save_format='keras')

# %%

# check to see that the model has weights and keys from History object
print(model.get_weights()[0][0][0][0])
print(model.history.history.keys())

# %%

# load the saved model

model_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\saved_models\\simple_classifier\\simple_classifier_1.keras"

simple_graph_classifier = tf.keras.models.load_model(model_path)

# %%

test_dataset = tf.data.Dataset.list_files("C:/Users/pat_h/OneDrive/Desktop/Graph Classifier/graph dataset/bar_resized/*.jpg")


# %%

# convert images from folder into arrays
evaluate_data = []

for i in test_dataset:
    with Image.open(tf.get_static_value(i).decode('utf-8')) as im:
        im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)
    evaluate_data.append(im_array)


# %%

# convert array into tensor and then into tensor dataset
tensor_data = map(tf.convert_to_tensor, evaluate_data)
tensor_data = list(tensor_data)
tensor_data = tf.data.Dataset.from_tensors(tensor_data)

# use loaded model to predict image class
pred_new = simple_graph_classifier.predict(tensor_data, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

# display first 20 predictions to confirm if it is working
pred_new = pd.DataFrame(pred_new, columns=['graph', 'natural'])
pred_new.head(n=20)


# %%

# load generated graphs to predict class, remember these are PNG images!
bar_files = tf.data.Dataset.list_files("C:/Users/pat_h/OneDrive/Desktop/Graph Classifier/generated_graphs/bar/*.png", shuffle=False)

# %%

for i in bar_files:
    print(tf.get_static_value(i).decode('utf-8'))
    break



# %%

evaluate_bar = []

for i in bar_files:
    with Image.open(tf.get_static_value(i).decode('utf-8')) as im:
        im = im.convert('RGB')
        im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)
    evaluate_bar.append(im_array)


# %%

print(evaluate_bar[0])
print(evaluate_bar[0].shape)

# %%

evaluate_bar = tf.image.resize(
    evaluate_bar,
    (IMG_HEIGHT, IMG_WIDTH),
    method=image.ResizeMethod.LANCZOS5,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)

# %%


tensor_bar = map(tf.convert_to_tensor, evaluate_bar)
tensor_bar = list(tensor_bar)
tensor_bar = tf.data.Dataset.from_tensors(tensor_bar)

# %%

bar_predict = simple_graph_classifier.predict(tensor_bar, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)


# %%

bar_predict = pd.DataFrame(bar_predict, columns=['graph', 'natural'])
bar_predict.head(n=20)


# %%

bar_predict['correct'] = bar_predict['graph'] > bar_predict['natural']
bar_predict.head(n=20)



# %%
counts = bar_predict['correct'].value_counts()

# %%

print(counts)


# %%
