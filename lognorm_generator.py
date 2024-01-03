# https://matplotlib.org/stable/gallery/statistics/histogram_cumulative.html#sphx-glr-gallery-statistics-histogram-cumulative-py
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

#%%

import numpy as np
import matplotlib.pyplot as plt
import random as random
import os

from lorem_ipsum_prep import lorumipsum as text

# %%

rng = np.random.default_rng()
mu, sigma = 3., 1. # mean and standard deviation
s = rng.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

x = np.linspace(min(bins), max(bins))
pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')
plt.show()

# %%

print(x.shape)
print(pdf.shape)

# %%

### generates and saves highly randomized graphs of the log-normal distribution ###

# use lorem ipsum to avoid language bias
words = np.array(text)
rng = np.random.default_rng()

rand_rgb = [] # randomized rgb
rgb_bin = [] # bin color of histogram
rgb_line = [] # color of lines
rgb_face = [] # face color of figure
rgb_ax = [] # inside color of figure

rgb_white = (1,1,1)
rgb_black = (0,0,0)
graph_amount = 50

for graph in range(graph_amount):
    
    # reset color lists for each iteration
    rand_rgb = []
    rgb_bin = []
    rgb_line = []
    rgb_face = []
    rgb_ax = []

    fig, ax = plt.subplots()

    # example data
    mu = rng.integers(-3.5,3.5) * rng.random()  # mean of distribution
    sigma = rng.integers(1,2.5) * rng.random()  # standard deviation of distribution
    s = rng.lognormal(mu, sigma, 420)

    num_bins = rng.integers(15,60)

    # randomize color of bins
    for i in range(3):

        color = round(random.random(), 1)
        rgb_bin.append(color)

    # add alpha 0 to hide bins 50% of the time
    rand_int = random.random()
    if rand_int > 0.5:
        rgb_bin.append(0)

    # the histogram of the data
    n, bins, ignored = plt.hist(s, num_bins, density=True) # align='mid'

    # add a 'best fit' line
    x = np.linspace(min(bins), max(bins), 1000)
    y = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))

    for i in range(3):

        color = round(random.random(), 1)
        rgb_line.append(color)

    if rng.random() > 0.5:
        line_style = '-'
    else:
        line_style = '--'

    ax.plot(x, y, line_style, color=rgb_line, linewidth=rng.random()*3)

    # fig face color
    for i in range(3):

        color = round(random.random(), 1)
        rgb_face.append(color)

    if rng.random() > 0.6: # create more white graphs
        rgb_face = rgb_white

    fig.set_facecolor(rgb_face)

    # ax face color
    for i in range(3):

        color = round(random.random(), 1)
        rgb_ax.append(color)
    
    if rng.random() > 0.6:
        rgb_ax = rgb_white

    ax.set_facecolor(rgb_ax)

    # set lorum ipsum labels
    ax.set_ylabel(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)))
    ax.set_xlabel(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)))
    ax.set_title(' '.join(np.random.choice(text, size = rng.integers(2,5), replace=False)))

    # add grid 50% of the time
    if rng.random() > 0.5:
        plt.grid(color='k', linestyle='-', linewidth=rng.random())

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    fig.savefig(os.path.join('test_graphs', 'lognorm', f'lognorm_{graph}.jpg'))
    plt.close(fig) # save memory usage

# %%
