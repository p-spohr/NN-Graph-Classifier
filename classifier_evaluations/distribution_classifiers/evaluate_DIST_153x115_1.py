# %%

import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd

from tensorflow import image
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' ## Stops kernal error bug with matplotlib and tensorflow


# %%

# model to predict
MODEL_FILE_NAME = "DIST_153x115_1.keras"

load_model_file_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\saved_models\\distribution_graph_classifiers\\{MODEL_FILE_NAME}"

print(load_model_file_path)

saved_model = tf.keras.models.load_model(load_model_file_path)
BATCH_SIZE = 20 # for prediction later

if saved_model:
    print(f'Model loaded: {MODEL_FILE_NAME}')


# %%

# fill each key with list of image tensor arrays
dist_image_dict = {'exp':[], 'lognorm':[], 'norm':[], 'unif':[]}

eval_dataset_path = f"C:\\Users\\pat_h\\htw_berlin_datasets\\DIST_153x115_1_DATASET"

for dirpath, dirnames, filenames in os.walk(eval_dataset_path):

    print(dirpath)

    for file in filenames:

        for dist_type in dist_image_dict.keys():

            if os.path.basename(dirpath) == dist_type:
                
                with Image.open(os.path.join(dirpath, file)) as im:

                    im_array = keras.utils.img_to_array(im, data_format=None, dtype=None)

                    # this dictionary will remain with simple array for convert to image later
                    # im_array = tf.convert_to_tensor(im_array)

                dist_image_dict[dist_type].append((file, im_array))


# %%

# convert array into tensor and then into tensor dataset

dist_image_dict_tensors = {}

for key, value in dist_image_dict.items():

    # create separate dictionary only for tensor data
    dist_image_dict_tensors[key] = []

    for file, array in value:

        tensor_array = tf.convert_to_tensor(array)

        dist_image_dict_tensors[key].append(tensor_array)
    
    # convert list of tensor data into TensorDataset
    dist_image_dict_tensors[key] = tf.data.Dataset.from_tensors(dist_image_dict_tensors[key])

     
print(dist_image_dict_tensors.keys())
print(type(dist_image_dict_tensors['exp']))

# %%

##### use loaded model to predict image class #####

pred_dict = {}

for key, value in dist_image_dict_tensors.items():
   
    pred_dict[key] = saved_model.predict(value, batch_size=BATCH_SIZE, verbose="auto", steps=None, callbacks=None)

    pred_dict[key] = pd.DataFrame(pred_dict[key], columns=['exp', 'lognorm', 'norm', 'unif']) # direct from model's train_ds.class_names


# %%

print(pred_dict['exp'].head(n=20))
print(type(pred_dict['exp']))
print(pred_dict['exp'].max(axis=1))

# %%

accuracy_check_dict = {}

for key, value in pred_dict.items():

    pred_dict[key] = value.assign(prediction=value.max(axis=1) == value[key])

    accuracy_check_dict[key] = pred_dict[key].prediction.value_counts()


# %%
    
print(pred_dict['unif'].head(n=10))

# %%

false_pred = 0 
false_count = []

for key, value in accuracy_check_dict.items():
    print(f'--------{key}--------')
    false_pred += value.loc[False]
    false_count.append(value.loc[False])
    value.loc['Accuracy'] = value.loc[True] / (value.loc[True] + value.loc[False]) 
    print(value.head())
    


print(false_pred)
print(false_count)
# %%

print(list(accuracy_check_dict.keys()))

# %%

false_count_df = pd.DataFrame(np.atleast_2d(false_count), columns=list(accuracy_check_dict.keys()))
false_count_df.head()
false_count_df.to_csv('eval_distribution_false_count.csv')


# %%

accuracy_check_df = pd.DataFrame(accuracy_check)
accuracy_check_df.head()
accuracy_check_df.to_csv(f'{EVAL_IMAGE_FOLDER}_in_{MODEL_FILE_NAME}_predictions.csv')

# %%

print(f'Accuracy: {100 * accuracy_check[True]/(accuracy_check[True] + accuracy_check[False])} %') 

# %%


# %%

# create new tuple list so I can sort and save the files easier
eval_image_pred = list(zip([file for file, array in eval_image_dataset], [array for file, array in eval_image_dataset], pred_new['prediction']))
print(type(eval_image_pred))

# %%

print(len(eval_image_pred[0]))

# %%

for file, image, prediction in eval_image_pred:
    print(file)
    print(image.shape)
    print(prediction)
    break


# %%

save_path = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\false_images\\generated"

for file, image, prediction in eval_image_pred:
    if prediction == False:
        with keras.utils.array_to_img(image) as im_false:
            im_false.save(os.path.join(save_path, file))


# %%

##### count the instances of falsely labeled graphs #####
DATASET = 'generated'
count_false_file_path = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\false_images\\{DATASET}"

dist_types_dict = {'exp':0, 'lognorm':0, 'norm':0, 'unif':0}

for dirpath, dirnames, filenames in os.walk(count_false_file_path):
    
    print(dirpath)

    for file in filenames:

        split_file = file.split('_')

        for key, value in dist_types_dict.items():

            if split_file[0] == key:

                dist_types_dict[key] += 1


for key, value in dist_types_dict.items():
    print(f'{key}: {value}')

# %%

new_df = pd.DataFrame.from_dict(dist_types_dict, orient='index', columns=['Count'])
new_df.head()
new_df.to_csv(f'eval_{DATASET}_false_count.csv')

# %%
