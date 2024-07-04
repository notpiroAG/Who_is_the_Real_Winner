# WHO IS THE REAL WINNER
This repository contains the code for the Kaggle competition Who is the real winner.
The task was to predict the education level of recent State and UT election winners across India for different states and UTs.
You can access more about the competitions here https://www.kaggle.com/competitions/who-is-the-real-winner

## PreProcessing and Models

### Data Preprocessing
I performed data preprocessing using a training-validation split technique to ensure robust model performance. This involved cleaning the dataset, handling missing values, and engineering relevant features.

### Models Used
To predict the education level of recent State and UT election winners across India, I opted for a Naive Bayes classifier. The decision to use Naive Bayes was based on its ability to handle categorical data effectively and its simplicity in implementation.

### Implementation Example
Below is a snippet of our Naive Bayes classifier implementation:

```python
# Training the Bernoulli Naive Bayes Classifier
nb_classifier = BernoulliNB()
nb_classifier.fit(X_train, y_train)

# Predicting on the validation set
y_pred_nb = nb_classifier.predict(X_valid)
#Calculatng f1_score
f1_score_nb = f1_score(y_valid, y_pred_nb, average='weighted')

