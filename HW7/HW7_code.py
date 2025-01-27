import pandas as pd
from matplotlib.pyplot import *
from pydtmc import *
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split
import numpy as np

#Naive Bayes
#Input Dateset
org_df = pd.read_csv("amr_ds.csv")

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'Not_MDR']
feat_df = org_df.loc[:,org_df.columns != 'Not_MDR']

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.25, random_state=1)

#Create Naive Bayes Model
nb_model = GaussianNB()  # CategoricalNB() for categorical data

nb_model.fit(x_train, np.ravel(y_train))

#Accuracy of Model
print("Naïve Bayes Test accuracy:  ", nb_model.score(x_test,y_test))


'''
Using all records in amr_ds.csv dataset to calculate following formula manually or using a piece of code:
amp_pen = number of records with Ampicillin=1 and Penicillin =1
amp_nmdr = number of records with Ampicillin=1 and Not_MDR =1
pen_nmdr = number of records with Penicillin =1 and Not_MDR =1
'''

amp_pen_df = org_df.query("Ampicillin == 1 & Penicillin == 1")
amp_nmdr_df = org_df.query("Ampicillin == 1 & Not_MDR == 1")
pen_nmdr_df = org_df.query("Penicillin == 1 & Not_MDR == 1")
amp_pen = amp_pen_df.shape[0]
amp_nmdr = amp_nmdr_df.shape[0]
pen_nmdr = pen_nmdr_df.shape[0]

states =  ['Ampicillin','Penicillin','Not_MDR']
transition_matrix = [[0, amp_pen/(amp_nmdr+amp_pen), amp_nmdr/(amp_nmdr+amp_pen)],
                                    [amp_pen/(pen_nmdr+amp_pen), 0, pen_nmdr/(pen_nmdr+amp_pen)],
                                    [amp_nmdr/(amp_nmdr+pen_nmdr), pen_nmdr/(amp_nmdr+pen_nmdr), 0]]
#print(transition_matrix)

# Create Markov Chain
mc = MarkovChain(transition_matrix, states)
print("stationary state:",mc.steady_states)

'''
Suppose that there is an association between infection after surgery and resistances 
to antimicrobials as follows:
        No Infection        Infection
Amp        0.4                 0.6
Pen        0.3                 0.7
NMDR       0.8                 0.2

Predict most probable sequence of states if we observe the following sequence in a patient:
[Infection after surgery, No infection after surgery, Infection after surgery]
'''
hidden_states = states
observation_symbols = ['NoInfection', 'Infection']
observation_matrix = [[0.4, 0.6],
                      [0.3, 0.7],
                      [0.8, 0.2]]

hmm = HiddenMarkovModel(transition_matrix, observation_matrix, hidden_states, observation_symbols)
lp, most_probable_states = hmm.predict(prediction_type='viterbi', symbols=['Infection','NoInfection','Infection'])
print("most probable sequenceis:",most_probable_states)
