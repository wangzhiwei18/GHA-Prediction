# GHA_Prediction

## Files
input.csv contains the raw, unprocessed dataset, where the data has not undergone preprocessing steps such as normalization, centering, or type conversion. It includes the values of various features corresponding to build, along with runID and repoID.

The data_cv.RData file stores the processed dataset, where the data has undergone preprocessing operations like normalization, centering, and type conversion, making it ready for direct use in model training.

The model.R file contains code for training and testing using 10-fold cross-validation repeated 10 times, employing algorithms like baseline, logistic regression (LR), k-nearest neighbors (KNN), decision trees (DT), Naive Bayes (NB), random forests (RF), and support vector machines (SVM). It also includes code for conducting ablation studies.

## Usage
To reproduce the results presented in the paper, simply run the model.R file.
