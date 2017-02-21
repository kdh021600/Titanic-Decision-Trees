# Titanic survival machine learning model
# Jason Dean
# Feb 20, 2017
# check out my website for more details:  jasontdean.com
  
# include this for Rmarkdown
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)
rm( list = ls())  # Clear environment
library(knitr)

# set seed for reproducibility
set.seed(12345)

# Data Cleanup
library(titanic)
data(titanic_train)
kable(head(titanic_train, 5), format="markdown", align = 'c')
tt.reduced <- titanic_train[,-c(1,4,9)]
kable(head(tt.reduced, 5), format="markdown", align = 'c')
sapply(tt.reduced, function(x){sum(is.na(x))})
sum(is.na(tt.reduced$Age))/nrow(tt.reduced)
sapply(tt.reduced, function(x){sum(x=='')})
tt.reduced <- tt.reduced[,-8]
tt.reduced <- tt.reduced[tt.reduced$Embarked != '',]
tt.reduced <- tt.reduced[is.na(tt.reduced$Age) == FALSE,]
sapply(tt.reduced, function(x){sum(is.na(x))})
nrow(tt.reduced)
kable(head(tt.reduced, 5), format="markdown", align = 'c')
tt.reduced$Survived <- as.factor(tt.reduced$Survived)
tt.reduced$Pclass <- as.factor(tt.reduced$Pclass)
tt.reduced$Sex <- as.factor(tt.reduced$Sex)
tt.reduced$Embarked <- as.factor(tt.reduced$Embarked)
str(tt.reduced)

# split data into training and test sets
splitMe <- sample(2, nrow(tt.reduced), replace = TRUE, prob = c(0.7,0.3))
tt.reduced.train <- tt.reduced[splitMe==1,]
tt.reduced.test <- tt.reduced[splitMe==2,]


# Data Exploration
summary(tt.reduced.train$Survived)
summary(tt.reduced.train$Sex)
table(tt.reduced.train$Survived, tt.reduced.train$Sex)

library(dplyr)
tt.reduced.train %>% group_by(Sex) %>% summarise(avgAge = mean(Age), stdev = sd(Age))

library(ggplot2)
ggplot(data=tt.reduced.train, aes(x=Age, fill=Sex)) + geom_density(alpha=0.3)

summary(tt.reduced.train$Fare)

ggplot(data=tt.reduced.train, aes(x=Fare)) + geom_histogram(binwidth = 5, col = "black", aes(y=..density..)) + theme_bw()

ggplot(tt.reduced.train, aes(x=Fare, y=Survived)) + geom_point() + theme_bw()

ggplot(tt.reduced.train, aes(x=Fare, y=Survived)) + geom_point() + theme_bw() + xlim(0,65)

# Decision Trees
library(rpart)
library(rattle)
library(rpart.plot)

tree.survival = rpart(Survived~., data=tt.reduced.train)
print(tree.survival$cptable)
fancyRpartPlot(tree.survival)
cp <- min(tree.survival$cptable[,1])
pruned.tree.survival <- prune(tree.survival, cp=cp)
tree.survival.predict <- predict(pruned.tree.survival, tt.reduced.test, type="class")

library(caret)
confusionMatrix(tree.survival.predict, tt.reduced.test$Survived)

library(randomForest)
rf.tree.survival <- randomForest(Survived~., data=tt.reduced.train)
plot(rf.tree.survival)
ntrees <- which.min(rf.tree.survival$err.rate[,1])
rf.tree.survival <- randomForest(Survived~., data=tt.reduced.train, ntree=ntrees)
rf.trees.predict <- predict(rf.tree.survival, tt.reduced.test, type="class")
confusionMatrix(rf.trees.predict, tt.reduced.test$Survived)

# Model Performance
require(ROCR)

tree.survival.predict2 <- predict(rf.tree.survival, tt.reduced.test, type="prob")
predROC <- prediction(tree.survival.predict2[,2], tt.reduced.test$Survived)
perfROC <- performance(predROC, "tpr", "fpr")
plot(perfROC)
abline(a=0, b=1)
perfROC <- performance(predROC, "auc")
perfROC@y.values[[1]]
