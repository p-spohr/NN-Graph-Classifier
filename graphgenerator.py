
# %%

from loremipsumprep import lorumipsum as text

print(text)

# %%

import numpy as np

tabcolors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan','tab:white'])

testwords = np.random.choice(text, size = 4, replace=False)
print(testwords)

print(' '.join(np.random.choice(text, size = np.random.randint(2,5), replace=False)))

# ' '.join(str(np.random.choice(text, size = np.random.randint(2,5), replace=False)))

randcol = np.random.choice(tabcolors, size = 1)[0]
print(randcol)
bar_colors = np.random.choice(tabcolors, size=4, replace=False)
print(bar_colors)

while randcol in bar_colors:
    randcol = np.random.choice(tabcolors, size = 1)[0]
print(randcol)

# %%

# generates highly randomized bar charts

import matplotlib.pyplot as plt
import numpy as np

tabcolors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan','white'])

# use lorem ipsum to avoid language bias
words = np.array(text)

fig, ax = plt.subplots()

element_count = np.random.randint(2,8)

words = np.random.choice(text, size = element_count, replace=False)
counts = [np.random.randint(0,500) for num in range(element_count)]

bar_labels = words
bar_colors = np.random.choice(tabcolors, size=element_count, replace=False)
randcol = np.random.choice(tabcolors, size = 1)[0]
while randcol in bar_colors:
    randcol = np.random.choice(tabcolors, size = 1)[0]

fig.set_facecolor(np.random.choice(tabcolors, size = 1)[0])

ax.bar(words, counts, label=bar_labels, color=bar_colors)
ax.set_facecolor(randcol)
ax.set_ylabel(' '.join(np.random.choice(text, size = np.random.randint(2,5), replace=False)))
ax.set_title(' '.join(np.random.choice(text, size = np.random.randint(2,5), replace=False)))

plt.show()


# %%


# generates highly randomized bar charts

import matplotlib.pyplot as plt
import numpy as np
import os

tabcolors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
             'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
             'tab:olive', 'tab:cyan','white'])

# use lorem ipsum to avoid language bias
words = np.array(text)

os.chdir('C:\\Users\\pat_h\\OneDrive\\Desktop\\Graph Classifier\\graphs')

for bar in range(100):
    fig, ax = plt.subplots()

    element_count = np.random.randint(2,8)

    words = np.random.choice(text, size = element_count, replace=False)
    counts = [np.random.randint(0,500) for num in range(element_count)]

    bar_labels = words
    bar_colors = np.random.choice(tabcolors, size=element_count, replace=False)
    randcol = np.random.choice(tabcolors, size = 1)[0]
    while randcol in bar_colors:
        randcol = np.random.choice(tabcolors, size = 1)[0]

    fig.set_facecolor(np.random.choice(tabcolors, size = 1)[0])

    ax.bar(words, counts, label=bar_labels, color=bar_colors)
    ax.set_facecolor(randcol)
    ax.set_ylabel(' '.join(np.random.choice(text, size = np.random.randint(2,5), replace=False)))
    ax.set_title(' '.join(np.random.choice(text, size = np.random.randint(2,5), replace=False)))

    fig.savefig(f'bar\\{bar}_bar.png')

# %%
