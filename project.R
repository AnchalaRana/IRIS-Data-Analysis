library(caret)
###### SETTING WORKING DIRECTORY #######
setwd("C:/Users/Anchala/Desktop/R_Studio_files")

######### READING DATA #############
iris <- read.csv("IRIS.csv" , header = TRUE)
str(iris)
View(iris)

# First we need to check that no datapoint is missing, 
#otherwise we need to fix the dataset.
apply(iris,2,function(x) sum(is.na(x)))

#### RANDOM SAMPLING OF DATASET #####
iris <- iris[sample(nrow(iris),150),]
View(iris)

standardiser <- function(x){
  (x-min(x))/(max(x)-min(x))
}

#### STANDARDISE  VALUES IN DATASET ###
iris[, 1:4] <- lapply(iris[, 1:4], standardiser)
View(iris)

##########################################
############ SUMMARIZE DATASET ###########
##########################################

#Dimensions of Dataset
dim(iris)
# list types for each attribute
sapply(iris, class)
# take a peek at the first 5 rows of the data
head(iris)
# list the levels for the class
levels(iris$species)
# summarize the class distribution
percentage <- prop.table(table(iris$species)) * 100
cbind(freq=table(iris$species), percentage=percentage)
# summarize attribute distributions
summary(iris)

##############################################
############# UNIVARIATE PLOTS ###############
##############################################

# split input and output
x <- iris[,1:4]
y <- iris[,5]
# boxplot for each attribute on one image
par(mfrow=c(1,4))
clr = c("orange" ,"green","blue","yellow")
for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i] , col = clr[i])
}

# barplot for class breakdown
plot(y , col= clr)

####################################################
############### MULTIVARIATE PLOTS #################
####################################################

# scatterplot matrix
featurePlot(x=x, y=y, plot="pairs" , auto.key=list(columns=3))

#Finding overlap between predictor and outcome/target variable 
# Sepal_Length
boxplot(iris$sepal_length~iris$species,data=iris, 
        main="Finding Overlap between predictor and outcome",   
        ylab="Species", xlab="Sepal_Length", horizontal=TRUE, col=terrain.colors(3))

# Petal_Width
boxplot(iris$petal_width~iris$species,data=iris, 
        main="Finding Overlap between predictor and outcome",   
        ylab="Species", xlab="Petal_width", horizontal=TRUE, col=terrain.colors(3))

# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box", col= iris$species)

# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales ,auto.key=list(columns=3))

library(ggthemes)
# Histogram
histogram <- ggplot(data=iris, aes(x=sepal_width)) +
  geom_histogram(binwidth=0.2, color="black", aes(fill=species)) + 
  xlab("Sepal Width") +  
  ylab("Frequency") + 
  ggtitle("Histogram of Sepal Width")+
  theme_economist()
print(histogram)

# Faceting: Producing multiple charts in one plot
library(ggthemes)
facet <- ggplot(data=iris, aes(sepal_length, y=sepal_width, color=species))+
  geom_point(aes(shape=species), size=1.5) + 
  geom_smooth(method="lm") +
  xlab("Sepal Length") +
  ylab("Sepal Width") +
  ggtitle("Faceting") +
  theme_fivethirtyeight() +
  facet_grid(. ~ species) # Along rows
print(facet)

## boundry between two predictors  z
library(ggplot2) 
qplot(iris$sepal_width,iris$sepal_length,data=iris,colour=species, size=3) 
qplot(iris$petal_length , iris$petal_width,data=iris,colour=species, size=3) 


######################################################
#################Correlation Analysis#################
######################################################

#It emphsize on what we say using box plot, It can tell if predictor is a good predictor or not a good predictor
#This analysis can help us decide if we can drop some columns/predictors depending upon its correlation with the outcome variable
library(psych)
pairs.panels(iris[, c(1:4,5)])



##################################################################
###############  DECISION TREE MODEL #############################
##################################################################

#Splitting of dataset 
set.seed(123)
dt = sort(sample(nrow(iris), nrow(iris)*.6))
train1<-iris[dt,]
tt<-iris[-dt,]
df = sort(sample(nrow(tt), nrow(tt)*.5))
validate1<-tt[df,]
test1<-tt[-df,]


library(rpart) 
library(rpart.plot) 
library(rattle) 
library(caret) 
dt_model<- rpart(species ~ . , data = train1) 
summary(dt_model) 
fancyRpartPlot(dt_model) 

predictions = predict(dt_model, test1, type = "class") 
table(predictions)
confusion.matrix = prop.table(table(predictions, test1$species)) 
confusion.matrix 
confusionMatrix(predictions,test1$species) 

#############################################################
################ RANDOM FOREST MODEL ########################
#############################################################

library(randomForest)
model <- randomForest(species ~ ., data=train1) 
model
#importance of each predictor 
importance(model) 
############ Testing Random forest ############ 
library(caret) 
predicted <- predict(model, test1) 
table(predicted) 
confusionMatrix(predicted, test1$species) 


#############################################################
########### BUILDING MODEL USING NEURAL NETWORK #############
#############################################################

library('ggplot2')
library('nnet')
library('dplyr')
library('reshape2')

# Convert your observation class and Species into one hot vector.
labels <- class.ind(as.factor(iris$species))

iris1 <- cbind(iris[,1:4], labels)
View(iris1)

# Splitting of dataset 
set.seed(123)
dt = sort(sample(nrow(iris1), nrow(iris1)*.6))
train<-iris1[dt,]
tt<-iris1[-dt,]
df = sort(sample(nrow(tt), nrow(tt)*.5))
validate<-tt[df,]
test<-tt[-df,]

#################Correlation Analysis#################
#It emphsize on what we say using box plot, It can tell if predictor is a good predictor or not a good predictor
#This analysis can help us decide if we can drop some columns/predictors depending upon its correlation with the outcome variable
library(psych)
pairs.panels(iris1[, c(1:4,5:7)])


# Training model using Neuralnet 
f <- as.formula("setosa + versicolor + virginica ~ sepal_length + sepal_width + petal_length + petal_width")

library('neuralnet')
iris_net <- neuralnet(f, data = train, hidden = c(16, 12), act.fct = "tanh", linear.output = FALSE)
plot(iris_net)

# Compute Predictions
pred <- compute(iris_net, test[,1:4])
# Extracting the Results
pred <- round(pred$net.result)
head(pred)

# Predicted Result
Result <- ifelse(pred[,1] %in% 1, 'setosa',
                 ifelse(pred[,2] %in% 1, 'versicolor', 'virginica'))
# Actual Result
Species <- ifelse(test[,5] %in% 1, 'setosa',
                  ifelse(test[,6] %in% 1, 'versicolor', 'virginica'))

# Confusion Matrix
mat = table(Species , Result)

# accuracy
accuracy <- (sum(diag(mat)) / sum(mat))
print(paste("Accuracy = " ,  accuracy))

# precision
precision <- (diag(mat) / rowSums(mat))
print(paste("Precision = " ,  precision))

# precision for particular class
precision.setosa <- precision["setosa"]
print(paste("Precision.Setosa = " ,  precision.setosa))
precision.versicolor <- precision["versicolor"]
print(paste("Precision.Versicolor = " ,  precision.versicolor))
precision.virginica <- precision["virginica"]
print(paste("Precision.Virginica = " ,  precision.virginica))

# recall for each class
recall <- (diag(mat) / colSums(mat))
print(paste("Recall = " ,  recall))



library(caret)
library(klaR)
data(iris)

# Define train control for k fold cross validation
train_control <- trainControl(method="cv", number=10)
# Fit Naive Bayes Model
model <- train(Species~., data=iris, trControl=train_control, method="nb")
# Summarise Results
print(model)