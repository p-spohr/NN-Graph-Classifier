
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()

# %%

import os
cwd = os.getcwd()
cwd
os.chdir('C:\\Users\\pat_h\\OneDrive\\Desktop')
cwd = os.getcwd()
cwd

# %%

os.getcwd()

# %%

fig


# %%

fig.savefig('firstbarchart.png', transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None
       )

# %%
import random
randcounts = []
for counts in range(3):
    randcounts.append(random.randint(0,100))

randcounts

# %%
fruits = ['apple', 'blueberry', 'cherry', 'orange', 'kiwi',
          'banana', 'strawberry', 'pineapple', 'watermelon', 'raspberry']

randcounts = [random.randint(0,100) for num in range(4)]
print(randcounts)
type(len(fruits))

# %%

import numpy as np

tabcolors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan'])
fruits = np.array(['apple', 'blueberry', 'cherry', 'orange', 'kiwi',
          'banana', 'strawberry', 'pineapple', 'watermelon', 'raspberry'])


targetfruits = np.random.choice(fruits, size=4, replace=False)
print(targetfruits)

# %%

import numpy as np

tabcolors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan']
fruits = ['apple', 'blueberry', 'cherry', 'orange', 'kiwi',
          'banana', 'strawberry', 'pineapple', 'watermelon', 'raspberry']


randfruits = [np.random.choice(fruits, replace=False) for num in range(4)]
randfruits

# %%

newlabel = 'barchart'
os.chdir('C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs')
os.makedirs(newlabel)
os.chdir(newlabel)
cwd = os.getcwd()
cwd


# %%

import numpy as np

tabcolors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan'])
fruits = np.array(['apple', 'blueberry', 'cherry', 'orange', 'kiwi',
          'banana', 'strawberry', 'pineapple', 'watermelon', 'raspberry'])

newlabel = 'barchart2'
os.chdir('C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs')
os.makedirs(newlabel)
os.chdir(newlabel)

for graph in range(100):

    fig, ax = plt.subplots()

    fruits = np.random.choice(fruits, size=4, replace=False)
    randnum = [random.randint(0,100) for num in range(4)]
    bar_colors = np.random.choice(tabcolors, size=4, replace=False)

    ax.bar(fruits, randnum, color=bar_colors)

    ax.set_ylabel('fruit supply')
    ax.set_title('Fruit supply by kind and color')

    fig.savefig(f'graph{graph}_barchart.png', transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None
       )
