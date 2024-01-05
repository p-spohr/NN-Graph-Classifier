# %%

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time



# %%

##### check folder and file names #####

work_path = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'

for dirpath, dirnames, filenames in os.walk(work_path):
    print('Top Folder: ' + os.path.basename(dirpath))
    for name in dirnames:
        print('\t' + 'Sub Folder: ' + name)
    for file in filenames[0:4]:
        print('\t' + 'File: ' + file + '\t' + str(os.path.getsize(os.path.join(dirpath, file))/1000) + ' KB')

# %%

##### show 3 x 3 image of graphs #####

# GRAPH_FOLDER = 'unif' 
# GRAPH_FOLDER = 'unif_32x32'
GRAPH_FOLDER = 'unif_115x86'

work_path = f'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\{GRAPH_FOLDER}'

plt.figure(figsize=(10,10))
count = 1
for dirpath, dirnames, filenames in os.walk(work_path):
    while count < 10:
        for file in filenames[0:9]:
            plt.subplot(3, 3, count)
            im = Image.open(os.path.join(work_path, file))
            plt.imshow(im)
            plt.axis('off')
            plt.title(str(file))
            count += 1


# %%

##### save all resized images into my 'all_32x32' folder #####
            
# create model of natural images vs all graphs from my graph_generators.py  
start_time = time.time()

work_path = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'
graph_types = ['exp', 'lognorm', 'norm', 'unif']

# walk through dir
for dirpath, dirnames, filenames in os.walk(work_path):
    print(f'PATH BASE: {os.path.basename(dirpath)}')

    # resize images and save to new dir 'resized'
    for file in filenames:
        if os.path.basename(dirpath) in graph_types:
            with Image.open(os.path.join(dirpath, file)) as im:
                im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                im.save(os.path.join(work_path, 'all_32x32', file))

end_time = time.time()
print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 64.904 seconds

# %%

##### ACCIDENTALLY SAVED THE WRONG FILE NAMES SO I NEEDED TO RENAME THEM ######
# rename all files to their correct distribution
start_time = time.time()

work_path = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'

# walk through dir
for dirpath, dirnames, filenames in os.walk(work_path):
    print(f'PATH BASE: {os.path.basename(dirpath)}')

    # resize images and save to new dir 'resized'
    for file in filenames:
        file_number = file.split('_')[1]
        new_name = f'{os.path.basename(dirpath)}_{file_number}'
        os.rename(os.path.join(dirpath, file), os.path.join(dirpath, new_name))

end_time = time.time()
print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 1.496 seconds

# %%

##### resized images and save in their own folders #####

# measure runtime
start_time = time.time()

# save folder location and list of graph distribution types
WORK_PATH = 'c:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets'
GRAPH_TYPES = ['exp', 'lognorm', 'norm', 'unif']

# image original dimension
ORIG_WIDTH_DIM = 460
ORIG_HEIGT_DIM = 345

# select number to scale the image down
SCALE_IMG = 3

# calculate new dimensions and configure file names
new_width_dim = round(ORIG_WIDTH_DIM / SCALE_IMG)
new_height_dim = round(ORIG_HEIGT_DIM / SCALE_IMG)
resized_dim = f'{new_width_dim}x{new_height_dim}'
graph_types_resized = [f'exp_{resized_dim}', f'lognorm_{resized_dim}', f'norm_{resized_dim}', f'unif_{resized_dim}']

print(new_width_dim,
      new_height_dim,
      resized_dim,
      graph_types_resized)

# walk through dir
for dirpath, dirnames, filenames in os.walk(WORK_PATH):
    print(f'PATH BASE: {os.path.basename(dirpath)}')

    # create new folders for the resized images
    for type in graph_types_resized:
        if dirpath == WORK_PATH and not os.path.exists(os.path.join(dirpath, type)):
            os.mkdir(os.path.join(dirpath, type))

    # resize images and save to new respective folders
    for file in filenames: # go through the list of files only once
        for type in GRAPH_TYPES:
            if os.path.basename(dirpath) == type:
                with Image.open(os.path.join(dirpath, file)) as im:
                    im = im.resize(size=(new_width_dim, new_height_dim), resample=Image.Resampling.LANCZOS)
                    im.save(os.path.join(WORK_PATH, f'{type}_{resized_dim}', file))

end_time = time.time()
print(f'Total Runtime: {round(end_time - start_time, 3)} seconds') # Total Runtime: 26.273 seconds
                
# %%
