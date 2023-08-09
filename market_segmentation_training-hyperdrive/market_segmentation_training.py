
# Import libraries
import argparse, joblib, os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Get the experiment run context
run = Run.get_context()

# Get script arguments
parser = argparse.ArgumentParser()

# Input dataset
parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')

# Hyperparameters
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100, help='number of estimators')
parser.add_argument('--max_depth', type=int, dest='max_depth', default=None, help='maximum depth of the tree')

# Add arguments to args collection
args = parser.parse_args()

# Log Hyperparameter values
run.log('n_estimators',  np.int(args.n_estimators))
run.log('max_depth',  np.float(args.max_depth) if args.max_depth else 'None')

# load the market segmentation dataset
print("Loading Data...")
segmentation_data = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels
X = segmentation_data.drop("Segmentation_A", "Segmentation_B", "Segmentation_C", "Segmentation_D", axis=1)
y = segmentation_data["Segmentation_A", "Segmentation_B", "Segmentation_C", "Segmentation_D"]

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a Random Forest classification model with the specified hyperparameters
print('Training a classification model')
model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth).fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# If the segmentation labels are binary (e.g., 0 and 1), we can compute AUC.
# If not, you'll need to adjust this or skip AUC computation.
if len(np.unique(y)) == 2:
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

# Save the model in the run outputs
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/market_segmentation_model.pkl')

run.complete()
