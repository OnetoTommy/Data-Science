import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
input_dataframe = pd.read_csv("Nutrition.csv")
print(input_dataframe)
##Smooth Noises using Bining
input_dataframe['protein_level'] = pd.qcut(input_dataframe['protein'], q=3)
input_dataframe['protein'] = pd.Series ([interval.mid for interval in
input_dataframe['protein_level']])
print(input_dataframe)
del input_dataframe['protein_level']
##Detect Outliers
q3, q1 = np.percentile(input_dataframe['calories'], [75 ,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_band,
'lower=',lower_band)
input_dataframe.loc[(input_dataframe['calories'] < lower_band) |
(input_dataframe['calories'] > upper_band), 'calories'] = None
print(input_dataframe)
##Encoding Categorical Variables
input_dataframe= pd.get_dummies(input_dataframe, dtype='int')
print(input_dataframe)
##MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(input_dataframe)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=input_dataframe.columns)
print(imputed_dataframe)
##Save Data
imputed_dataframe.to_csv("Nutrition_Cleaned.csv",index=False)
##Normalize data
norm_dataframe = (imputed_dataframe - imputed_dataframe.mean()) / imputed_dataframe.std()
print(norm_dataframe)
