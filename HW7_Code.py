import matplotlib.pyplot
import pandas as pd
from pydtmc import *
from sklearn.conftest import pyplot
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import accuracy_score
from sklearn.naive_bayes import *
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

###########################################
# Define probability of happening
###########################################
def get_number(df, column1, column2): ##probability of happening is short for poh

    numbers = df[(df[column1] == 1) & (df[column2] == 1)].shape[0]

    return numbers
###########################################

#input dataset
org_df =  pd.read_csv('amr_ds.csv')

#Define label and features
label_df = org_df['Not_MDR']
feat_df = org_df.loc[:, org_df.columns != "Not_MDR"]

#Split train and test data
train_x, test_x, train_y, test_y = train_test_split(feat_df, label_df, test_size=0.25, random_state=42)

#Create Naive Bayes model and calculate test accuracy
nb_model = GaussianNB()
nb_model.fit(train_x, train_y)
test_y_pred = nb_model.predict(test_x)
nb_accuracy = accuracy_score(test_y, test_y_pred)
print("Accuracy of Naive Bayes =", nb_accuracy)
# print(f'Test accuracy: {nb_model.score(test_x,test_y):.2f}')

#Count the number of 0 and 1
amp_pen = get_number(org_df, 'Ampicillin', 'Penicillin')
amp_nmdr = get_number(org_df, 'Ampicillin', 'Not_MDR')
pen_nmdr = get_number(org_df, 'Penicillin', 'Not_MDR')

# print(amp_pen)
# print(amp_nmdr)
# print(pen_nmdr)

# print('amp_pen =',amp_pen)
# print('amp_nmdr =',amp_nmdr)
# print('amp_nmdr =',amp_nmdr)

#Markov Chain
# The states
states = ["Ampicillin","Penicillin","Not_MDR"]

#Transition Matrix
transition_matrix = [
    [0, amp_pen/(amp_pen+amp_nmdr), amp_nmdr/(amp_pen+amp_nmdr)],
    [amp_pen/(pen_nmdr+amp_pen), 0, pen_nmdr/(amp_pen+pen_nmdr)],
    [amp_nmdr/(amp_nmdr+pen_nmdr), pen_nmdr/(pen_nmdr+amp_nmdr), 0],
]

#Create Markov Chain
mc = MarkovChain(transition_matrix, states)
print(mc)

#Show stationary state
print(mc.steady_states)

# Visualize results
plt.ion()
plt.figure(figsize=(10, 6))
plot_graph(mc)
plot_redistributions(mc, 70, plot_type='projection', initial_status='Ampicillin')
plt.show()

# Hidden Markov
hidden_states = ["Ampicillin", "Penicillin", "Not_MDR"]
observation_symbols = ['Infection', 'No Infection']
emission_matrix = [[0.4, 0.6], # AMP
                   [0.3, 0.7], # PEN
                   [0.8, 0.2]] # NMDR

# Create Hidden Markov Model
hmm = HiddenMarkovModel(transition_matrix, emission_matrix, hidden_states, observation_symbols)

# Visualize results
plt.ion()
plt.figure(figsize=(10, 10))
plot_graph(hmm)
plot_sequence(hmm, steps=10, plot_type='matrix')

# Predict hidden states
lp, most_probable_states = hmm.predict(prediction_type='viterbi', symbols=['Infection', 'No Infection', 'Infection'])
print(most_probable_states)
print(lp)




# # 1. Define the number of hidden states (Ampicillin, Penicillin, Not_MDR)
# n_states = 3
#
# # 2. Define the number of possible observations (Infection, No Infection)
# n_observations = 2
#
# # 3. Define the transition matrix
# transition_matrix = np.array([
#     [0.33, 0.33, 0.34],  # From Ampicillin
#     [0.33, 0.33, 0.34],  # From Penicillin
#     [0.33, 0.33, 0.34]   # From Not_MDR
# ])
#
# # 4. Define the emission probabilities as means and variances for continuous observation values
# means = np.array([[0.6], [0.7], [0.2]])  # Mean infection probabilities for each state
# covars = np.tile(np.identity(1), (n_states, 1, 1))  # Covariances (since we assume infection probability is a single value)
#
# # 5. Define the initial probabilities for each state
# start_probabilities = np.array([1/3, 1/3, 1/3])
#
# # 6. Create the HMM model
# model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
#
# # 7. Assign the start, transition, and emission probabilities
# model.startprob_ = start_probabilities
# model.transmat_ = transition_matrix
# model.means_ = means
# model.covars_ = covars
#
# # 8. Define the observations (Infection = 1, No Infection = 0)
# observations = np.array([[1], [0], [1]])  # Sequence: Infection, No Infection, Infection
#
# # 9. Run the Viterbi algorithm to find the most probable sequence of hidden states
# logprob, hidden_states = model.decode(observations, algorithm="viterbi")
#
# # 10. Print the most probable sequence of hidden states
# state_names = ['Ampicillin', 'Penicillin', 'Not_MDR']
# most_probable_states = [state_names[state] for state in hidden_states]
#
# print("Most probable states sequence:", most_probable_states)

