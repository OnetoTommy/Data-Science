from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_nb(X_train,y_train,X_test,y_test):
    print('*** Training Naive Bayes')

    # The Gaussian Naive Bayes algorithm can be used when the features are continuous. It assumes that the continuous features follow a normal (Gaussian) distribution.
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    #Visual inspection on feature X_train_pca_df to check if the data is normally distributed
    # Histogram: Plot a histogram of all the features in x_train_pca_df and overlay a normal distribution curve.
    '''
    X_train.hist( bins=30, figsize=(40,10))
    plt.show()
    '''
    '''
    # Histogram: Plot a histogram of the data and overlay a normal distribution curve.
    feature = X_train_pca_df['PC1']
    plt.hist(feature, bins=30, density=True, alpha=0.6, color='blue')
    mean, std = np.mean(feature), np.std(feature)
    x = np.linspace(min(feature), max(feature), 100)
    plt.plot(x, stats.norm.pdf(x, mean, std), color='red')  # Normal curve
    plt.title('Histogram with Normal Curve')
    plt.show()

    '''

    '''
    # Q-Q plot: Plot the quantiles of the data against the quantiles of a normal distribution.
    stats.probplot(X_train_pca_df['PC3'], dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()
    '''


    # Accuracy on train data
    train_pred_label = nb_model.predict(X_train)
    train_accuracy = accuracy_score(train_pred_label, y_train)
    print(f"Train Data Accuracy: {train_accuracy}")


    # we should be careful Zero Frequency Problem when using Naive Bayes,but for our case we don't have this problem
    predict = nb_model.predict(X_test)

    test_accuracy = accuracy_score(predict, y_test)
    print(f"Test Data Accuracy: {test_accuracy}")

    return nb_model