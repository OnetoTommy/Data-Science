import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)
# IQR method to remove outliers
def remove_outliers(dataframe, columns = "*", copy = False):
    df = dataframe.copy() if copy else dataframe
    upper = 75
    lower = 25
    if columns == "*" :
        columns = df.columns
    for column in columns :
        q3, q1 = np.nanpercentile(df[column], [upper, lower])
        fence = 1.5 * (q3 - q1)
        upper_band = q3 + fence
        lower_band = q1 - fence
        print('column:',column,'q1=', q1, ' q3=', q3, ' IQR=', q3 - q1, ' upper=', upper_band,
              'lower=', lower_band)

        print("value of " + column + " in these rows are out of the band")
        # print removed row
        print(df.loc[(df[column] < lower_band) |
                            (df[column] > upper_band), column])
        df.loc[(df[column] < lower_band) |
                            (df[column] > upper_band), column] = np.nan

    return df


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#using MICE to impute missing values. but we need to impute the missing values with the value that is not out of the band
def impute(dataframe):
    df = dataframe
    minimum_before = list(df.iloc[:, :].min(axis=0))
    maximum_after = list(df.iloc[:, :].max(axis=0))
    im = IterativeImputer(max_iter=10, random_state=0,min_value=minimum_before, max_value=maximum_after)
    im_df = im.fit_transform(df)
    im_df = pd.DataFrame(im_df, columns=df.columns)

    # Show the imputed values in the dataframe
    imputed_values = im_df.where(pd.isna(df)).dropna(how='all')
    print("Imputed Values:")
    print(imputed_values)

    return im_df
def categorize(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    return df
def print_nan(df):
    rows_with_nan = df[df.isna().any(axis=1)]
    print(rows_with_nan)

