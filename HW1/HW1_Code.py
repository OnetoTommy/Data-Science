# Develop a code using a common programing language (Python, R, Java, C#, C++, etc.)
# over the dataset to remove outliers from BMI column and impute all missing values
# in the dataset.

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
input_dataframe = pd.read_csv("InsuranceCharges.csv")
#print(input_dataframe)

##Detect Outliers
q3, q1 = np.nanpercentile(input_dataframe['bmi'], [75 ,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_band,
'lower=',lower_band)
input_dataframe.loc[(input_dataframe['bmi'] < lower_band) |
(input_dataframe['bmi'] > upper_band), 'bmi'] = None
print(input_dataframe)

##MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(input_dataframe)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=input_dataframe.columns)
print(imputed_dataframe)

##Save Data
imputed_dataframe.to_csv("HW1_CleanedDataset.csv",index=False)
