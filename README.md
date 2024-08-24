# GHA-BFP

## Files
input.csv contains the raw, unprocessed dataset, where the data has not undergone preprocessing steps such as normalization, centering, or type conversion. It includes the values of various features corresponding to build, along with runID and repoID.

The data_time.RData file stores the processed dataset, where the data has undergone preprocessing operations like normalization, centering, and type conversion, making it ready for direct use in model training.

The model.R file contains code for training and testing using time-series-validation, employing algorithms like baseline, logistic regression (LR), k-nearest neighbors (KNN), decision trees (DT), Naive Bayes (NB), random forests (RF), and support vector machines (SVM). It also includes code for conducting ablation studies.

## Usage
To reproduce the results presented in the paper, simply run the model.R file.

## Demo
The demo of GHA-BFP is available at https://huggingface.co/spaces/GHA-study/GHA-BFP_Demo. To use the demo, you need to optimize the following steps:
- Input the information of the build.
- Click the "predict" button.
- Wait for the result to be revealed in the "prediction result" textbox.
