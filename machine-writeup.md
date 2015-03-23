---
title: "Machine writeup"
author: "Gulraiz Shabbir"
date: "Sunday, March 22, 2015"
output: html_document
---

## Summary
Here is a prediction algorithm which is approximately 93% accurate. The prediction model used for the purpose is random forest.
This project will use accelerometer data from six participants to predict their level of performance within the parameters of the collected data. The accelerometer were on the belt, forearm, arm, and dumbell.


#### Libraries

The following libraries may be used in the project code:


```r
library(caret)
library(kernlab)
library(randomForest)
library(corrplot)
library(knitr)
library(ggplot2)
```

#### Data: Loading and Processing

The data were downloaded from an Amazon cloudfront. The two csv files contain the training and test data, and were put into a directory.

Download the data:


```r
# Create directory, if it does not exist
#if (!file.exists("pmlData")) {
#dir.create("pmlData")
#}

# Fetch file and save to destination directory
#fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Download file from internet
# add method="curl" if on a mac or linux box
#download.file(fileUrl1, , destfile = "./pmlData/pml_training.csv")
#download.file(fileUrl2, , destfile = "./pmlData/pml_testing.csv") 
#dateDownloaded <- date()
```

Loading the training data:


```r
# Read the csv file for initial inspection
dataPML_Train <- read.csv("./pmlData/pml_training.csv", na.strings= c("NA",""," "))

# clean the data by removing NAs
dataNARmv <- apply(dataPML_Train, 2, function(x) {sum(is.na(x))})
dataTrain <- dataPML_Train[,which(dataNARmv == 0)]

# remove identifier columns
dataTrain <- dataTrain[8:length(dataTrain)]
```

#### Create the Predictive Model

The test data set was divided into training and cross validation sets in a 70:30 ratio to train the model and then test it against unfitted data.


```r
# divide data into training and cross-validation sets
inTrain <- createDataPartition(y = dataTrain$classe, p = 0.7, list = FALSE)
training <- dataTrain[inTrain, ]
crossValid <- dataTrain[-inTrain, ]
```


A random forest model was selected to predict the classification. It had methods for error correction in unbalanced data sets. The correlation between any two trees in the forest increases the forest error rate. Therefore, a correllation plot was produced in order to see how strong the variable relationships were.



```r
# plot the correlation matrix
corMatrix <- cor(training[, -length(training)])
corrplot(corMatrix, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 


In this type of plot the dark red and blue colours represent a high negative and positive relationship, respectively, between the variables. Highly correlated predictors are fine, so they may all be included in the model.

A model was then fitted with the outcome set of the training class and all other variables were used in the prediction.



```r
# fit a model to predict the classe using everything a predictor
model <- randomForest(classe ~ ., data = training)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.52%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B   15 2640    3    0    0 0.0067720090
## C    0   14 2377    5    0 0.0079298831
## D    0    0   22 2228    2 0.0106571936
## E    0    0    3    5 2517 0.0031683168
```


The model produced an OOB error rate of .56%. This was considered to be acceptable, therefore the model was applied to the test data.


#### Cross-Validation of the Data

The model was used to classify the remaining 30% of data. The results were output to a confusion matrix, and included the actual classifications to ascertain model accuracy.


```r
# test the model using the remaining 30% of data
predictXValid <- predict(model, crossValid)
confusionMatrix(crossValid$classe, predictXValid)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    1 1137    1    0    0
##          C    0    4 1020    2    0
##          D    0    0   11  952    1
##          E    0    0    2    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9943, 0.9977)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9965   0.9865   0.9979   0.9991
## Specificity            1.0000   0.9996   0.9988   0.9976   0.9996
## Pos Pred Value         1.0000   0.9982   0.9942   0.9876   0.9982
## Neg Pred Value         0.9998   0.9992   0.9971   0.9996   0.9998
## Prevalence             0.2846   0.1939   0.1757   0.1621   0.1837
## Detection Rate         0.2845   0.1932   0.1733   0.1618   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9980   0.9926   0.9977   0.9993
```

The model accuracy was 99.3%. This model can now be used to predict new data.


#### The Predictions

A separate data set was then loaded into R and cleaned in the same manner as before. The model was then used to predict the classifications of the 20 results of this new data.



```r
# apply the same treatment to the final testing data
dataPML_Test <- read.csv("./pmlData/pml_testing.csv", na.strings= c("NA",""," "))
dataTestNArmv <- apply(dataPML_Test, 2, function(x) {sum(is.na(x))})
dataTest <- dataPML_Test[,which(dataTestNArmv == 0)]
dataTest <- dataTest[8:length(dataTest)]

# predict the classes of the test set
predictTest <- predict(model, dataTest)
predictTest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


#### Conclusion

With the abundance of data provided from multiple measurement sources, it is possible to accurately predict, within reason, how well a person is preforming an excercise using a relatively simple model.

