import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

# defining some functions for preprocessing
def digit_currency(x):
    if isinstance(x, str):
        if 'Crore+' in x:
            return float(x.replace(' Crore+', '')) * 10000000
        elif 'Lac+' in x:
            return float(x.replace(' Lac+', '')) * 100000
        elif 'Thou+' in x:
            return float(x.replace(' Thou+', '')) * 1000
        elif 'Hund+' in x:
            return float(x.replace(' Hund+', '')) * 1000
        else:
            return float(x.replace('$', '').replace(',', ''))
    return x


# Loading the datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# seperating features(train) and labels(y) of training data
y = train['Education']
y = y.to_frame()
train.drop(['Education'],axis = 1,inplace = True)


# making currency in digits 
train['Total Assets'] = train['Total Assets'].apply(digit_currency)
train['Liabilities'] = train['Liabilities'].apply(digit_currency)

# Removing unnecessary columns
train.drop(['ID', 'Candidate', 'Constituency ∇'], axis=1, inplace=True)


# Normalizing the numerical features
scaler = MinMaxScaler()
train[['Total Assets', 'Liabilities','Criminal Case']] = scaler.fit_transform(train[['Total Assets', 'Liabilities','Criminal Case']])

t_train = pd.get_dummies(train, columns=['Party','state'])

t_train.columns = t_train.columns.astype(str)
t_train = t_train.astype(float)




# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(t_train, y, test_size=0.25, random_state=50)

# # Train the RandomForestClassifier
# rf_classifier = RandomForestClassifier(n_neighbors=10)
# rf_classifier.fit(X_train, y_train)

# # Predict on the validation set
# y_pred_rf = rf_classifier.predict(X_valid)
# f1_score_rf = accuracy_score(y_valid, y_pred_rf)
# print("F1 Score on validation set:", f1_score_rf)
# print("Classification Report:")
# print(classification_report(y_valid, y_pred_rf))



# Train the Bernoulli Naive Bayes Classifier
nb_classifier = BernoulliNB()
nb_classifier.fit(X_train, y_train)

# Predict on the validation set
y_pred_nb = nb_classifier.predict(X_valid)
f1_score_nb = f1_score(y_valid, y_pred_nb, average='weighted')
print("F1 Score on validation set:", f1_score_nb)
print("Classification Report:")
print(classification_report(y_valid, y_pred_nb))

# Preprocessing the test set

test.drop(['ID', 'Candidate', 'Constituency ∇'], axis=1, inplace=True)

test['Total Assets'] = test['Total Assets'].apply(digit_currency)
test['Liabilities'] = test['Liabilities'].apply(digit_currency)

t_test = pd.get_dummies(test, columns=['Party','state'])

t_test.columns = t_test.columns.astype(str)
t_test = t_test.astype(float)

# Predicting education levels for the test set using Naive Bayes
test_predictions_nb = nb_classifier.predict(t_test)

# Creating submission dataframe for Naive Bayes
index_values = np.arange(len(test_predictions_nb))
predictions_df = pd.DataFrame({'ID': index_values, 'Education': test_predictions_nb})

predictions_df.to_csv('220065_submission_bn_0.25_check.csv', index=False)

