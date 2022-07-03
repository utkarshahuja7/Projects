## ----setup, include=FALSE, warning=FALSE-----------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## --------------------------------------------------------------------------------------------------------------------------
# Load the packages

library(xgboost)
library(caret)
library(tidyverse)
library(tidymodels)
library(tibble)
library(randomForest)
library(here)
library(gbm)


## --------------------------------------------------------------------------------------------------------------------------
# Read data

tr <- read_csv(here::here("mouse_tr.csv")) %>%
  mutate(celltype = factor(celltype))
ts <- read_csv(here::here("mouse_ts_mask.csv"))
ts_sample<- read_csv(here::here("mouse_ts_samp.csv"))

celltype <- tr$celltype

# merging data sets to add celltypes to the test set
ts<- merge(x=ts,y=ts_sample, by="location", all.x = TRUE)



## --------------------------------------------------------------------------------------------------------------------------
# Fitting the xgboost model with cv = 10

mouse_xg <- train(celltype ~., data = tr[,-1],
                  method= "xgbTree",
                  trControl= trainControl("cv", number = 10))



## --------------------------------------------------------------------------------------------------------------------------
# Determining the best tune for the model

mouse_xg$bestTune



## --------------------------------------------------------------------------------------------------------------------------
# Conversion of the labels for training set

trainData_label <- as.integer(tr$celltype)-1

# setting null value to the column in tr

tr$celltype <- NULL

# Conversion of the labels for test set

testData_label <- as.integer(factor(ts$celltype))-1

# setting null value to the column in ts

ts$celltype <- NULL



## --------------------------------------------------------------------------------------------------------------------------
# Creating the training and testing sets for the model

train_data <-  as.matrix(tr[,-1])

test_data <-  as.matrix(ts[,-1])



## --------------------------------------------------------------------------------------------------------------------------
# Creating the xgb.DMatrix objects

dtrain <-  xgb.DMatrix(data=train_data, label= trainData_label)
dtest <-  xgb.DMatrix(data=test_data, label= testData_label)




## --------------------------------------------------------------------------------------------------------------------------
# Defining the main parameters for the classification

num_class = length(levels(celltype))
params = list(
  booster="gbtree",
  eta=0.3,
  max_depth=1,
  gamma=0,
  subsample=0.75,
  colsample_bytree=0.8,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class = num_class
)



## --------------------------------------------------------------------------------------------------------------------------
# Training the model for classification

dfit <- xgb.train(
  params=params,
  data= dtrain,
  nrounds=150,
  nthreads=1,
  watchlist=list(train=dtrain, test=dtest),
  verbose=0
)

# Examine the model fit

dfit



## --------------------------------------------------------------------------------------------------------------------------
# Predicting the results with the test data for celltypes

pred_test <-  predict(dfit,test_data,reshape = T)
pred_test <-  as.data.frame(pred_test)
colnames(pred_test) = levels(celltype)



## --------------------------------------------------------------------------------------------------------------------------
# Identifying the predicted labels that has the best chance of being true

pred_celltype = apply(pred_test,1,function(x) colnames(pred_test)[which.max(x)])


mouse_pred <-  data.frame(ts, celltype = pred_celltype)



## --------------------------------------------------------------------------------------------------------------------------
# extracting the predictions in .csv format

write_csv(mouse_pred[,c(1, 1005)], file="mypred_final.csv")


## --------------------------------------------------------------------------------------------------------------------------
# References-

#XGBoost Parameters — xgboost 1.6.1 documentation. (2022). Retrieved 27 May 2022, from https://xgboost.readthedocs.io/en/stable/parameter.html

#Sarafian, R. (2022). Tree based models. Retrieved 27 May 2022, from https://rstudio-pubs-static.s3.amazonaws.com/408996_debd530b72c844db9be5ef6ef9febc54.html

#Kube, D. (2022). XGBoost Multinomial Classification Iris Example in R. Retrieved 27 May 2022, from https://rstudio-pubs-static.s3.amazonaws.com/456044_9c275b0718a64e6286751bb7c60ae42a.html

#Malshe, A. (2022). 2.4 XGBoost | Data Analytics Applications. Retrieved 27 May 2022, from https://ashgreat.github.io/analyticsAppBook/xgboost

