# %%

from os import getcwd, chdir, mkdir, walk, rename, replace, remove
from os.path import join, exists, split, basename, getsize
from shutil import copyfile, move, copytree
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cwd = getcwd()
print(cwd)


# %%

for dirpath, dirnames, filenames in walk('c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs'):
    print('Top Folder: ' + basename(dirpath))
    for name in dirnames:
        print('\t' + 'Sub Folder: ' + name)
    for file in filenames[0:10]:
        print('\t' + 'File: ' + file + '\t' + str(getsize(join(dirpath, file))/1000) + ' KB')

# %%


for dirpath, dirnames, filenames in walk(cwd):
    print(dirpath)
    if basename(dirpath) == 'bar':
        print('GOT EM!')
        for file in filenames:
            print(f'{file} copied to {join(cwd, "bar2", file)}')
            copyfile(join(dirpath, file), join(cwd, "bar2", file))
    else:
        print('Nope!')
    


# %%

mkdir(join(cwd, 'bar2'))

for dirpath, dirnames, filenames in walk(cwd):
    for name in dirnames: 
        if name == 'bar':
            for file in filenames[0:10]:
                copyfile(join(dirpath, file), join(cwd, 'bar2'))


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs\\bar2'

for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames:
        rename(join(workpath, file), join(workpath, '000' + file))



# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs\\bar2'

for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames[0:10]:
        print(file)



# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'

graph_types = ['bar', 'diagram', 'flow', 'graph', 'pie', 'table']

graph_types_dict = {'bar':0, 'diagram':0, 'flow':0, 'graph':0, 'pie':0, 'table':0}

for key, value in graph_types_dict.items():
    print(f'{key} : {value}')


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
count = 0
for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames:
        for key, value in graph_types_dict.items():
            if key in file:
                graph_types_dict[key] += 1

for key, value in graph_types_dict.items():
    print(f'{key} : {value}')



# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'

for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames[0:10]:
        file_split = file.split('_')
        print(file_split)
# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'

plt.figure(figsize=(10,10))
count = 1
for dirpath, dirnames, filenames in walk(workpath):
    while count < 10:
        for file in filenames[0:9]:
            plt.subplot(3, 3, count)
            im = Image.open(join(workpath, file))
            plt.imshow(im)
            plt.axis('off')
            count += 1

# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'

fig = plt.figure(figsize=(10,10))
count = 1
for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames[10:19]:
        fig.add_subplot(3, 3, count)
        im = Image.open(join(workpath, file))
        plt.imshow(im)
        plt.axis('off')
        count += 1

# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'

graph_types = ['bar', 'diagram', 'graph', 'pie',]

for dirpath, dirnames, filenames in tqdm(walk(workpath)):
    for type in tqdm(graph_types):
        mkdir(join(workpath, type))
        for file in tqdm(filenames):
            split_file = file.split('_')
            if type == split_file[0]:
                copyfile(src=join(workpath, file), dst=join(workpath,type,file))


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types = ['bar', 'diagram', 'graph', 'pie',]

images = []

for dirpath, dirnames, filenames in walk(join(workpath)):
    print(basename(dirpath))
    for file in filenames[0:3]:
        print('\t' + join(dirpath, file))
        im = Image.open(join(dirpath, file))
        images.append(im)

        
# %%

rescaled_images = []

for image in images:
    image = image.resize((32,32), resample=Image.Resampling.LANCZOS)
    print(image.size)
    rescaled_images.append(image)
    

# %%

image_array = np.array(rescaled_images[0])

print(image_array.shape)

pil_image = Image.fromarray(image_array, mode='RGB')

pil_image.show()


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
newfolder = 'rawdata'

if not exists(join(workpath, newfolder)):
    mkdir(join(workpath, newfolder))

for dirpath, dirnames, filenames in walk(workpath):
    for file in filenames:
        if basename(dirpath) == 'graph dataset':
            move(src=join(dirpath,file), dst=join(workpath, newfolder, file))


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
for dirpath, dirnames, filenames in walk(workpath):
    print(f'Path: {basename(dirpath)}')
    for dirname in dirnames:
        print(f'\tFolder: {dirname}')
    for file in filenames[0:3]:
        print(f'\tFile: {file}')


# %%

# Copy all files into new folders to be resized
workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types = ['bar', 'diagram', 'graph', 'pie',]

for dirpath, dirnames, filenames in walk(workpath):
    for dirname in dirnames:
        if dirname in graph_types:
            print('SCR: ' + join(dirpath, dirname))
            print('\tDST: ' + join(dirpath, f'{dirname}_resized'))
            copytree(join(dirpath, dirname), join(dirpath, f'{dirname}_resized'))


# %%

def resize_3232(image_path):
    with Image.open(image_path) as im:
        im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
    return im
# %%

print(getcwd())
# map() requires an interable object
bar_chart = ['barchart_test.png']
bar_chart_resized = map(resize_3232, bar_chart)

# map returns a map object that needs to be recasted to index form with list()
bar_chart_resized = list(bar_chart_resized)
bar_chart_resized[0].show()


# %%

# Using map would require listing each file, performing map, and then listing them again to copy. I'm already iterating through the files so a map() is redundant.

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]

for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')
    for file in filenames:
        if basename(dirpath) in graph_types_resized:
            with Image.open(join(dirpath, file)) as im:
                im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                im.save(join(dirpath, f'r_{file}'))


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]


for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')
    for file in filenames:
        if basename(dirpath) in graph_types_resized:
            if file.split('_')[0] != 'r':
                remove(join(dirpath, file))



    


# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

# walk through dir
for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')
    # this isn't working it's only resizing the files in bar and it is making 4 extra folders in each graph_types folder
    # make dir for resized images
    
    for type in graph_types_resized:
        mkdir(join(dirpath, type))

    # resize images and save to new dir
        for file in filenames:
            if basename(dirpath) in graph_types:
                with Image.open(join(dirpath, file)) as im:
                    im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                    im.save(join(workpath, type, file))



# %%

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')

    # make dir for resized images
    # for type in graph_types_resized:
    #     mkdir(join(dirpath, type))

    # resize images and save to new dir
    for file in filenames:
        
        if basename(dirpath) in graph_types:

            with Image.open(join(dirpath, file)) as im:
                im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                im.save(join(workpath, 'resized', file))



# %%


workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

# walk through dir
for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')
    # this isn't working it's only resizing the files in bar and it is making 4 extra folders in each graph_types folder
    # make dir for resized images
    if dirpath == workpath:
        for type in graph_types_resized:
            mkdir(join(dirpath, type))

    # resize images and save to new dir
    # this resizes and saves all files to each new folder, works well if I want to put them all into one folder
    for type in graph_types_resized:
        for file in filenames:
            if basename(dirpath) in graph_types:
                with Image.open(join(dirpath, file)) as im:
                    im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                    im.save(join(workpath, type, file))


# %%

# save all resized images into my 'resized' folder
###### WORKS ######

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

# walk through dir
for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')

    # resize images and save to new dir 'resized'
    for file in filenames:
        if basename(dirpath) in graph_types:
            with Image.open(join(dirpath, file)) as im:
                im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                im.save(join(workpath, 'resized', file))



# %%

# I am determined to find a way to save the resized images in their own folders
###### WORKS ######

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

# walk through dir
for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')

    # create new folders for the resized images
    if dirpath == workpath:
        for type in graph_types_resized:
            mkdir(join(dirpath, type))

    # resize images and save to new respective folders
    for file in filenames: # go through the list of files only once
        for type in graph_types:
            if basename(dirpath) == type:
                with Image.open(join(dirpath, file)) as im:
                    im = im.resize(size=(32,32), resample=Image.Resampling.LANCZOS)
                    im.save(join(workpath, f'{type}_resized', file))


                
# %%

# double check the sizes of the new images
###### WORKS ######

workpath = 'c:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graph dataset'
graph_types_resized = ['bar_resized', 'diagram_resized', 'graph_resized', 'pie_resized',]
graph_types = ['bar', 'diagram', 'graph', 'pie',]

# walk through dir
for dirpath, dirnames, filenames in walk(workpath):
    print(f'PATH BASE: {basename(dirpath)}')

    # double check the sizes of the new images
    for file in filenames[0:3]: # go through the list of files only once
        for type in graph_types_resized:
            if basename(dirpath) == type:
                with Image.open(join(dirpath, file)) as im:
                    print(im.size)
# %%


