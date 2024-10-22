import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt

# Input dataset
org_df = pd.read_csv('amr_horse_ds.csv')

# Smooth dataset in binning
org_df['Age_binned'] = pd.qcut(org_df['Age'], 2)
# org_df['Age'] = pd.Series([interval.mid for interval in org_df['Age_binned']])
del org_df['Age']
print(org_df)

# Encoding for org_df
org_df = pd.get_dummies(org_df)
# categorical_columns = org_df.select_dtypes(include=['object', 'category']).columns
# org_df = pd.get_dummies(org_df, columns=categorical_columns)


# Hyperpramater for min_sup, min_conf, min_lift
min_sups = [0.05, 0.1, 0.4]
min_confs = [0.70, 0.85, 0.95]
min_lifts = [1.1, 1.5, 4]

#Extract the best association rule
num_rows = {}
best_num_rows = 0
best_min_sup = 0
best_min_conf = 0
best_min_lift = 0
i = 0

for min_sup in min_sups:
    for min_conf in min_confs:
        for min_lift in min_lifts:
            # Get frequent patterns using fpgrowth
            frequent_patterns_df = fpgrowth(org_df, min_support=min_sup, use_colnames=True)

            # Generate association rules
            rules_df = association_rules(frequent_patterns_df, metric="confidence", min_threshold=min_conf)

            # Filter rules based on lift
            lift_rules_df = rules_df[rules_df['lift'] > min_lift]

            # Store the number of rows for each combination
            num_rows[i] = len(lift_rules_df)

            # Check if the number of rules is between 20 and 50
            if 20 < num_rows[i] < 50:
                best_num_rows = num_rows[i]
                best_min_conf = min_conf
                best_min_sup = min_sup
                best_min_lift = min_lift
            i += 1
print('Extract rules =',best_num_rows)
print('Best_sup =', best_min_sup,
      'Best_conf =', best_min_conf,
      'Best_lift =', best_min_lift)

#Best association rule
frequent_patterns_df = fpgrowth(org_df, min_support = 0.1,use_colnames=True)
rules_df = association_rules(frequent_patterns_df, metric = "confidence", min_threshold = 0.95)
lift_rules_df = rules_df[rules_df['lift'] > 4]

#Save Association Rules
lift_rules_df.to_csv('arules_frequency.csv')

#Visualize Association Rules
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(lift_rules_df['support'], lift_rules_df['confidence'], lift_rules_df['lift'], marker="*")
ax.set_xlabel('support')
ax.set_ylabel('confidence')
ax.set_zlabel('lift')
# ax.set_xlim([0.1, max(rules_df['support'])])  # X 轴从 0.1 开始
# ax.set_ylim([0.95, max(rules_df['confidence'])])  # Y 轴从 0.95 开始
# ax.set_zlim([4, max(rules_df['lift'])])  # Z 轴从 4 开始
plt.show()