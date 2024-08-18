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

load("YourPath/data_time.RData")


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
# time-series-validation
fitControl <- trainControl(
  method = "timeslice",  
  initialWindow = 20000,    
  horizon = 3000,          
  fixedWindow = TRUE,    
  summaryFunction = model_summary,
  returnResamp = "final"
)

#parallel
set.seed(123)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# time-series-validation
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
SVML_benchmark=train(trainx,trainy,method="svmRadial",trControl=fitControl)
print(SVM_benchmark$resample)
print(SVM_benchmark)

#RF
set.seed(123)
Rf_benchmark=train(osx,osy,method="rf",trControl=fitControl)
print(Rf_benchmark$resample)
print(Rf_benchmark)

#NB
set.seed(123)
NB_benchmark=train(osx,osy,method="nb",trControl=fitControl)
print(NB_benchmark$resample)
print(NB_benchmark)

#DT
set.seed(123)
DT_benchmark=train(osx,osy,method="rpart",trControl=fitControl)
print(DT_benchmark$resample)
print(DT_benchmark)

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

# importance of the features
importance <- varImp(Rf_benchmark, scale = FALSE)
print(importance$importance)
plot(importance, maxrows = Inf)
