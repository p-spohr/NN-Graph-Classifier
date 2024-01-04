# https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from lorem_ipsum_prep import lorumipsum as text

# %%

### generates and saves highly randomized graphs of the normal distribution ###

# measure script runtime
start_time = time.time()

# use lorem ipsum to avoid language bias
words = np.array(text)
rng = np.random.default_rng(12345)

rand_rgb = [] # randomized rgb
rgb_bin = [] # bin color of histogram
rgb_line = [] # color of lines
rgb_face = [] # face color of figure
rgb_ax = [] # inside color of figure
rgb_xlabel = [] # color of text x label
rgb_ylabel = [] # color of text y label
rgb_title = [] # color of text title

RGB_WHITE = (1,1,1)
RGB_BLACK = (0,0,0)
GRAPHS = 100 # 2000 graphs: 305.011 seconds

# select save directory
SAVE_PATH = "C:\\Users\\pat_h\\htw_berlin_datasets\\dist_datasets\\norm"
# SAVE_PATH = "C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\test_graphs\\norm"
SAVE_DPI = 72
SAVE_FORMAT = 'jpg'


for graph in range(GRAPHS):
    
    # reset color lists for each iteration
    rand_rgb = []
    rgb_bin = []
    rgb_line = []
    rgb_face = []
    rgb_ax = []
    rgb_xlabel = []
    rgb_ylabel = []
    rgb_title = []

    fig, ax = plt.subplots()

    # randomize paramters
    mu = rng.integers(-80,80)  # mean of distribution
    sigma = rng.integers(1,50)  # standard deviation of distribution
    x = rng.normal(loc=mu, scale=sigma, size=rng.integers(50,2000))

    num_bins = rng.integers(30,120) # for histogram

    # randomize color of bins
    for i in range(3):

        color = round(rng.random(), 1)
        rgb_bin.append(color)

    # add alpha 0 to hide bins 50% of the time (only function line is visible)
    rand_int = rng.random()
    if rand_int > 0.5:
        rgb_bin.append(0)

    # the histogram of the data
    count, bins, patches = ax.hist(x, num_bins, density=True, color=rgb_bin)

    # add a 'best fit' line
    y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2)

    for i in range(3):
        color = round(rng.random(), 1)
        rgb_line.append(color)

    if rng.random() > 0.5:
        line_style = '-'
    else:
        line_style = '--'

    ax.plot(bins, y, line_style, color=rgb_line, linewidth=rng.random()*3)
   
    # fig face color
    for i in range(3):
        color = round(rng.random(), 1)
        rgb_face.append(color)
    if rgb_face == RGB_BLACK:
        rgb_face == RGB_WHITE
    if rng.random() > 0.6: # create more white graphs
        rgb_face = RGB_WHITE

    fig.set_facecolor(rgb_face)

    # ax face color
    for i in range(3):
        color = round(rng.random(), 1)
        rgb_ax.append(color)
    
    if rng.random() > 0.6:
        rgb_ax = RGB_WHITE

    ax.set_facecolor(rgb_ax)

    # x label rgb
    for i in range(3):
        color = round(rng.random(), 1)
        rgb_xlabel.append(color)
    # y label rgb
    for i in range(3):
        color = round(rng.random(), 1)
        rgb_ylabel.append(color)
    # title rgb
    for i in range(3):
        color = round(rng.random(), 1)
        rgb_title.append(color)

    # set lorum ipsum labels
    ax.set_ylabel(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)), color=rgb_ylabel, fontsize=max(10, rng.integers(20,30)*rng.random()))
    ax.set_xlabel(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)), color=rgb_xlabel, fontsize=max(10, rng.integers(20,30)*rng.random()))
    ax.set_title(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)), color=rgb_title, fontsize=max(10, rng.integers(20,30)*rng.random()))

    # add grid 50% of the time
    if rng.random() > 0.5:
        plt.grid(color='k', linestyle='-', linewidth=rng.random())

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    fig.savefig(os.path.join(SAVE_PATH, f'norm_{graph}.jpg'))
    plt.close(fig) # save memory usage


end_time = time.time()
print(f'Runtime: {round(end_time - start_time, 3)} seconds')

# %%

