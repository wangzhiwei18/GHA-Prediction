library(caret)
library(randomForest)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(doParallel)
library(Metrics)
library(deepnet)
library(pROC)
library(ROSE)
library(effsize)

load("YourPath/data_cv.RData")
# str(osx)
# str(osy)

#calculate baseline
set.seed(123)
folds <- createMultiFolds(osy, k=10, times=10)
# str(fold1)
# initiate metrics
accuracy_list <- numeric(length(folds))
precision_list <- numeric(length(folds))
recall_list <- numeric(length(folds))
f1_list <- numeric(length(folds))

databl <- data.frame(osy)
str(databl)

# 10 times 10-fold cv
for (i in 1:length(folds)) {
  # get index
  testIndex <- folds[[i]]
  
  # divide train & test set
  testData <- databl[testIndex, ]
  testData <- data.frame(testData)
  str(testData)
  
  # 50% random prediction
  random_predictions <- sample(levels(testData$testData), size = nrow(testData), replace = TRUE)
  
  # confusion matrix
  confusionMat <- confusionMatrix(as.factor(random_predictions), testData$testData)
  
  # get metrics
  accuracy <- confusionMat$overall["Accuracy"]
  precision <- confusionMat$byClass["Precision"]
  recall <- confusionMat$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  
  # 0 division
  if (is.nan(precision) || is.nan(recall) || is.nan(f1)) {
    precision <- 0
    recall <- 0
    f1 <- 0
  }
  
  # save results
  accuracy_list[i] <- accuracy
  precision_list[i] <- precision
  recall_list[i] <- recall
  f1_list[i] <- f1
}

# average metrics
mean_accuracy <- mean(accuracy_list)
mean_precision <- mean(precision_list)
mean_recall <- mean(recall_list)
mean_f1 <- mean(f1_list)

# print
cat("Mean Accuracy:", mean_accuracy, "\n")
cat("Mean Precision:", mean_precision, "\n")
cat("Mean Recall:", mean_recall, "\n")
cat("Mean F1 Score:", mean_f1, "\n")





model_summary <- function(data,lev=NULL,model=NULL) {
  
  confusion_matrix <- confusionMatrix(data$pred, data$obs)
  precision <- confusion_matrix$byClass["Precision"]
  accuracy <- confusion_matrix$overall["Accuracy"]
  recall <- confusion_matrix$byClass["Sensitivity"]
  f1_score <- (2*precision*recall) / (precision+recall)
  c(
    Accuracy = accuracy,
    Precision = precision,
    Recall = recall,
    F1 = f1_score
  )
  
}
fitControl=trainControl(method="repeatedcv", number=10, repeats=10, summaryFunction = model_summary, returnResamp = "final")

# table(osy)

#parallel
set.seed(123)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)



#LR
set.seed(123) 
LR_benchmark=train(osx,osy,method="glm",trControl=fitControl)
print(LR_benchmark$resample)
print(LR_benchmark)


#KNN
set.seed(123)
KNN_benchmark=train(osx,osy,method="knn",trControl=fitControl)
print(KNN_benchmark$resample)
print(KNN_benchmark)


#SVM
set.seed(123)
SVML_benchmark=train(trainx,trainy,method="svmLinear",trControl=fitControl)
print(SVML_benchmark)

#RF
set.seed(123)
Rf_benchmark=train(osx,osy,method="rf",trControl=fitControl)
print(Rf_benchmark$resample)
print(Rf_benchmark)

#NB
set.seed(123)
Rf_benchmark=train(osx,osy,method="nb",trControl=fitControl)
print(Rf_benchmark$resample)
print(Rf_benchmark)

#DT
set.seed(123)
Rf_benchmark=train(osx,osy,method="rpart",trControl=fitControl)
print(Rf_benchmark$resample)
print(Rf_benchmark)

###ablation experiment
#current_build
set.seed(123)
Rf1_benchmark=train(only1,osy,method="rf",trControl=fitControl)
print(Rf1_benchmark$resample)
print(Rf1_benchmark)
#historical_build
set.seed(123)
Rf2_benchmark=train(only2,osy,method="rf",trControl=fitControl)
print(Rf2_benchmark$resample)
print(Rf2_benchmark)
#configuration_file
set.seed(123)
Rf3_benchmark=train(only3,osy,method="rf",trControl=fitControl)
print(Rf3_benchmark$resample)
print(Rf3_benchmark)
#repository
set.seed(123)
Rf4_benchmark=train(only4,osy,method="rf",trControl=fitControl)
print(Rf4_benchmark$resample)
print(Rf4_benchmark)


wilcox.test(Rf_benchmark$resample$Accuracy.Accuracy,Rf1_benchmark$resample$Accuracy.Accuracy,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Accuracy.Accuracy,Rf1_benchmark$resample$Accuracy.Accuracy)

wilcox.test(Rf_benchmark$resample$Precision.Precision,Rf1_benchmark$resample$Precision.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Precision.Precision,Rf1_benchmark$resample$Precision.Precision)

wilcox.test(Rf_benchmark$resample$Recall.Sensitivity,Rf1_benchmark$resample$Recall.Sensitivity,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Recall.Sensitivity,Rf1_benchmark$resample$Recall.Sensitivity)

wilcox.test(Rf_benchmark$resample$F1.Precision,Rf1_benchmark$resample$F1.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$F1.Precision,Rf1_benchmark$resample$F1.Precision)

wilcox.test(Rf_benchmark$resample$Accuracy.Accuracy,Rf2_benchmark$resample$Accuracy.Accuracy,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Accuracy.Accuracy,Rf2_benchmark$resample$Accuracy.Accuracy)

wilcox.test(Rf_benchmark$resample$Precision.Precision,Rf2_benchmark$resample$Precision.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Precision.Precision,Rf2_benchmark$resample$Precision.Precision)

wilcox.test(Rf_benchmark$resample$Recall.Sensitivity,Rf2_benchmark$resample$Recall.Sensitivity,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Recall.Sensitivity,Rf2_benchmark$resample$Recall.Sensitivity)

wilcox.test(Rf_benchmark$resample$F1.Precision,Rf2_benchmark$resample$F1.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$F1.Precision,Rf2_benchmark$resample$F1.Precision)

wilcox.test(Rf_benchmark$resample$Accuracy.Accuracy,Rf3_benchmark$resample$Accuracy.Accuracy,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Accuracy.Accuracy,Rf3_benchmark$resample$Accuracy.Accuracy)

wilcox.test(Rf_benchmark$resample$Precision.Precision,Rf3_benchmark$resample$Precision.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Precision.Precision,Rf3_benchmark$resample$Precision.Precision)

wilcox.test(Rf_benchmark$resample$Recall.Sensitivity,Rf3_benchmark$resample$Recall.Sensitivity,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Recall.Sensitivity,Rf3_benchmark$resample$Recall.Sensitivity)

wilcox.test(Rf_benchmark$resample$F1.Precision,Rf3_benchmark$resample$F1.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$F1.Precision,Rf3_benchmark$resample$F1.Precision)

wilcox.test(Rf_benchmark$resample$Accuracy.Accuracy,Rf4_benchmark$resample$Accuracy.Accuracy,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Accuracy.Accuracy,Rf4_benchmark$resample$Accuracy.Accuracy)

wilcox.test(Rf_benchmark$resample$Precision.Precision,Rf4_benchmark$resample$Precision.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Precision.Precision,Rf4_benchmark$resample$Precision.Precision)

wilcox.test(Rf_benchmark$resample$Recall.Sensitivity,Rf4_benchmark$resample$Recall.Sensitivity,paired = TRUE)
cliff.delta(Rf_benchmark$resample$Recall.Sensitivity,Rf4_benchmark$resample$Recall.Sensitivity)

wilcox.test(Rf_benchmark$resample$F1.Precision,Rf4_benchmark$resample$F1.Precision,paired = TRUE)
cliff.delta(Rf_benchmark$resample$F1.Precision,Rf4_benchmark$resample$F1.Precision)