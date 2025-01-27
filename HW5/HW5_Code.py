from math import isnan

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from mlxtend.frequent_patterns import fpgrowth,apriori,association_rules

def describe_array (arr):
    df_describe = pd.DataFrame(arr)
    print(df_describe.describe())

#Input Dateset
org_df = pd.read_csv("amr_horse_ds.csv")
df = org_df.loc[:]

#there is some value we need to fix
df['Sex'] = df['Sex'].str.strip()
df['Sex'] = df['Sex'].replace({"CM":"MC"})
df['Gram_ID'] = df['Gram_ID'].map({"Positive":True, "Negative":False})

age_values = df['Age'].values.reshape(-1, 1)
# Apply K-Means with n clusters for binning
age_clusters_num = 2
kmeans = KMeans(n_clusters=age_clusters_num, random_state=42)
df['AgeBin'] = kmeans.fit_predict(age_values)

for age_bin in range(age_clusters_num):
    df_age_bin = df.loc[ df['AgeBin'] == age_bin ]
    bin_name = str(df_age_bin['Age'].values.min())+"-"+str(df_age_bin['Age'].values.max())
    df.replace({"AgeBin":age_bin},bin_name, inplace = True)
    print("Age Bin Name:", bin_name, " Count:", df_age_bin.shape[0])

df.drop("Age", axis=1, inplace=True)

org_df= pd.get_dummies(df)

min_sups = [0.05, 0.1, 0.4]
min_confs= [0.70, 0.85, 0.95]
min_lifts= [1.1, 1.5, 4]

#Extract Association Rules
for min_sup in min_sups:
    for min_conf in min_confs:
        for min_lift in min_lifts:
            frequent_patterns_df = fpgrowth(org_df, min_support=min_sup,use_colnames=True)
            rules_df = association_rules(frequent_patterns_df, metric = "confidence", min_threshold = min_conf)
            high_lift_rules_df = rules_df[rules_df['lift'] > min_lift]
            if 20 <= high_lift_rules_df.shape[0] <= 50:
                print({
                    'min_sup':min_sup,
                    'min_conf':min_conf,
                    'min_lift':min_lift
                })
                print("total rules:",high_lift_rules_df.shape[0])
                high_lift_rules_df.to_csv("HW5_Rules.csv")
                exit()
