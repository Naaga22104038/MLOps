## Import required libraries

## warnings
import warnings

warnings.filterwarnings("ignore")

## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## libraries for classification
from sklearn.pipeline import Pipeline
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

## for saving model
import pickle

## for word embedding with Spacy
import spacy
import en_core_web_lg

"""## Loading the dataset:"""

def parse_args():

    parser = argparse.ArgumentParser(description="Process input arguments")

    parser.add_argument(
	    "--preprocessed_data",
        type=str,
        dest="preprocessed_data",
        required = True   
        )
       
    return parser.parse_args()
    
args = parse_args()

df_all = pd.read_csv(args.preprocessed_data)

def conf_matrix_acc(y_true, y_pred):
  ## Plot confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
  plt.yticks(rotation=0)
  print("=========================================")
  print(f'Accuracy score is : {accuracy_score(y_true, y_pred)}')
  print("=========================================")
  print("Detail:")
  print(skm.classification_report(y_true, y_pred))

## Plot ROC and precision-recall curve
def roc_precision_auc():
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ## Plot roc
  for i in range(len(classes)):
      fpr, tpr, thresholds = skm.roc_curve(y_test_array[:,i],
                            probs[:,i])
      ax[0].plot(fpr, tpr, lw=3,
                label='{0} (area={1:0.2f})'.format(classes[i],
                                skm.auc(fpr, tpr))
                 )
  ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
  ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05],
            xlabel='False Positive Rate',
            ylabel="True Positive Rate (Recall)",
            title="Receiver operating characteristic")
  ax[0].legend(loc="lower right")
  ax[0].grid(True)

  ## Plot precision-recall curve
  for i in range(len(classes)):
    precision, recall, thresholds = skm.precision_recall_curve(
                y_test_array[:,i], probs[:,i])
    ax[1].plot(recall, precision, lw=3,
               label='{0} (area={1:0.2f})'.format(classes[i],
                                  skm.auc(recall, precision))
              )
  ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall',
            ylabel="Precision", title="Precision-Recall curve")
  ax[1].legend(loc="best")
  ax[1].grid(True)
  plt.show()
  print(f'AUC score is : {skm.roc_auc_score(Y_test, probs[:,1])}')

"""## Support Vector Machine(SVM) with word embedding:"""

nlp = en_core_web_lg.load()

## word-embedding
all_vectors = pd.np.array([pd.np.array([token.vector for token in nlp(s)]).mean(axis=0) * pd.np.ones((300)) \
                           for s in df_all['clean_text']])

# split out validation dataset for the end
Y = df_all["label"]
X = all_vectors

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

validation_size = 0.3
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=None)

# test options for classification
num_folds = 10
seed = 7
scoring = 'accuracy'

# Create a svm Classifier
clf = SVC(probability=True)

## Running the svm Classifier

# Full Training period
res = clf.fit(X_train, Y_train)
train_result = accuracy_score(res.predict(X_train), Y_train)
test_result = accuracy_score(res.predict(X_test), Y_test)

print("train_result:", "test_result:", train_result, test_result, sep=" ")

## Test results
##
y_pred_svm = res.predict(X_test)
classes = np.unique(Y_test.to_list())
y_test_array = pd.get_dummies(Y_test, drop_first=False).values
probs = res.predict_proba(X_test)
print (conf_matrix_acc(Y_test.to_list(), y_pred_svm))
print (roc_precision_auc())
