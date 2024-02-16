# %%

import pandas as pd


# %%

evaluations = pd.read_csv("..\\classifier_evaluations\\distribution_test_evaluation\\test_evaluate_DIST_153x115_1.csv", index_col=0)


# %%

evaluations.loc[evaluations['prediction']==False].to_csv('test_evaluate_false_DIST_153x115_1.csv')

# %%
