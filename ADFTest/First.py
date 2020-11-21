import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import sys

file_path = "C:\DATA\Projects\TutorialsOnGitHub\Statistics\ADFTest\NonStationary.xlsx"
#file_path = "C:\DATA\Projects\TutorialsOnGitHub\Statistics\ADFTest\Stationary.xlsx"


df = pd.read_excel(file_path)

df.head()

plt.plot(df["Value"])
plt.show()
X = df["Value"].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")

# Take the first discrete difference of the time series.
df["Diff_Value"] = df["Value"].diff()
diff_values = df["Diff_Value"].values

# Define the target variable as the difference column with nan values removed. 
Y = diff_values[~np.isnan(diff_values)] 

ts_values_orig = df["Value"].values

# Drop the last element of the feature so that the number of elements in the feature and target space are the same.
ts_values = ts_values_orig[:-1]

# An intercept is not included by default and should be added by the user
X = sm.add_constant(ts_values)

model = sm.OLS(Y,X)
results = model.fit()

print(results.summary())

# More details on the outputs of the OLS
# F-statistics: https://statisticsbyjim.com/regression/interpret-f-test-overall-significance-regression/#comment-7861
# Relation between standard error and R2: https://statisticsbyjim.com/regression/standard-error-regression-vs-r-squared/
# Relation between standard error and R2: https://www.youtube.com/watch?v=r-txC-dpI-E&ab_channel=statisticsfun
# https://www.youtube.com/watch?v=VvlqA-iO2HA&ab_channel=zedstatistics
