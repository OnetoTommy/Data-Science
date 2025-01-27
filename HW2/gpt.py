# Importing necessary libraries for decision tree and data processing
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
file_path = "diabetes.csv"  # Path to the dataset
diabetes_df = pd.read_csv(file_path)

# Define features and label
X = diabetes_df.drop(columns=['Outcome'])
y = diabetes_df['Outcome']

# Split the data into train, validation, and test sets (72%, 20%, 8%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.28, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2857, random_state=42, stratify=y_temp)

# Hyperparameter grid for tuning the Decision Tree
param_grid = {
    'max_depth': [3, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [3, 5],
    'min_impurity_decrease': [0.01, 0.001],
    'ccp_alpha': [0.001, 0.0001]
}

# Initialize the Decision Tree Classifier with Gini index
dtree = DecisionTreeClassifier(criterion='gini', random_state=42)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model based on validation data
best_tree = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred_test = best_tree.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(test_accuracy)

# Extract rules from the decision tree
tree_rules = export_text(best_tree, feature_names=list(X.columns))

# Displaying results
best_tree_params = grid_search.best_params_

print(best_tree_params)

print(tree_rules)

