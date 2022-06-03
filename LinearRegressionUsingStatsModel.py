import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

boston = datasets.load_boston()
X_train = boston['data']
y_train = boston['target']
￼

X_train_with_constant = sm.add_constant(X_train)
mod1 = sm.OLS(y_train, X_train_with_constant)
sm_fit1 = mod1.fit()
sm_predictions1 = sm_fit1.predict(X_train_with_constant)

df = pd.DataFrame(X_train, columns = boston['feature_names'])
df['target'] = y_train
display(df.head())

formula = 'target ~ ' + ' + '.join(boston['feature_names'])
print('formula:', formula)


mod2 = smf.ols(formula, data = df)
sm_fit2 = mod2.fit()
sm_predictions2 = sm_fit2.predict(df)

