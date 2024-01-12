# %%

import matplotlib.pyplot as plt
import pandas as pd

import os


# %%

saved_evaluations = f"C:\\Users\\pat_h\\OneDrive\\Desktop\\public-repos\\NN-Graph-Classifier\\classifier_evaluations\\distribution_classifiers"

eval_false_count = pd.read_csv(os.path.join(saved_evaluations, f'eval_distribution_false_count.csv'), index_col=0)
eval_false_count.head()
list(eval_false_count.columns)
eval_false_count.loc[0]

# %%


fig, ax = plt.subplots(1,1)
plt.bar(list(eval_false_count.columns), eval_false_count.loc[0])
plt.title('Misclassified Distribution Graphs in DIST_153x115_1')
plt.xlabel('Distributions')
plt.ylabel('Counts')
fig.savefig('Misclassified Distribution Graphs in DIST_153x115_1')


# %%
