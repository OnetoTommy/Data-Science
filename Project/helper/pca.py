import pandas as pd
import numpy as np
# Use PCA on the training data to create 3 new components from existing features (all columns except outcome).
# Transfer training and test data to the new dimensions (PCs).
from sklearn.decomposition import PCA

def PCA_extract(X_train, X_test, y_train, y_test, n_components=3):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    columns = []
    for i in range(n_components):
        columns.append("PC" + str(i+1))

    X_train_pca_df = pd.DataFrame(X_train_pca, columns=columns)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=columns)

    y_train_pca = np.ravel(y_train.astype(int))
    y_test_pca = np.ravel(y_test.astype(int))

    y_train_pca_df = pd.DataFrame(y_train_pca, columns=["Outcome"])
    y_test_pca_df = pd.DataFrame(y_test_pca, columns=["Outcome"])

    #combine X_train_pca and X_test_pca(only for check)
    '''
    X_train_pca_df["Outcome"] = y_train_pca_df
    X_test_pca_df["Outcome"] = y_test_pca_df
    combined_pca_df = pd.concat([X_train_pca_df, X_test_pca_df])
    combined_pca_df.to_csv("pcv.csv",index=False)
    covariance_matrix = pca.get_covariance()
    pc_feature_relationship = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2', 'PC3'])
    '''

    # Plot the proportion of explained variance by Cumulative sum of principal components
    '''x_axis = range(pca.n_components_)
    plt.plot(x_axis, np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel('Principal component')
    plt.ylabel('Cumulative sum of explained variance')
    plt.xticks(x_axis)
    plt.show()
    '''
    # Show eigenvalues.Kaiser’s rule:  principal components are retained, whose eigenvalue exceed 1.
    print("Eigenvalues:")
    print(pca.explained_variance_)


    return X_train_pca_df, X_test_pca_df, y_train_pca, y_test_pca