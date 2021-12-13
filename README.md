# multiclass-perceptron
This is a simple implementation of a multiclass perceptron (MP) to recognize digits.\
The algorithm is an extension version of the binary perceptron: https://github.com/rnels12/digit-perceptron.
An example of training data can be obtained from kaggle: https://www.kaggle.com/c/digit-recognizer/data.
But, any dataset with the same format as the one from kaggle can also be processed.
Furthermore, the trained model from the MP is compared with the one from the random forest (RF) of sklearn 
(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) .\
The result summary:\
The accuracy is about 86% for the MP model against 96% for the RF model (using the default setup).
However, the training time is much faster for the MP model than for the RF one, to be more precise, 
with the current setups, training the MP model is about 36 times faster than training the RF one.
Lastly, when used to predict the test dataset from kaggle, the model will yield an accuracy of about 87%, 
which is not bad for a homemade classifier.
