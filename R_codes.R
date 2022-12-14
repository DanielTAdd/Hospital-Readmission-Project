### Tetteh Daniel Addokwei 
# Intelligent Data Analytics 
# Homework 7

install.packages("glmnet")
install.packages("lubridate")
install.packages("tidymodels")
install.packages("outliers")
install.packages("DAAG")
install.packages("party")
install.packages("tree")
install.packages("MICE")
install.packages("RevoScaleR")
install.packages("ROCR")
install.packages("rattle")
install.packages("kernlab")
install.packages('MARS')
install.packages('binaryLogic')
install.packages('knn')
install.packages("MLeval")

library(xgboost)
library(randomForest)
library(knn)
library(earth)
library(kernlab)
library(RevoScaleR)
library(rpart.plot)
library(dplyr)
library(tidyverse)
library(readr)
library(VIM)
library(plotly)
library(corrplot)
library(caret)
library(car)
library(AppliedPredictiveModeling)
library(dplyr)
library(MASS)
library(lars)
library(elasticnet)
library(reshape)
library(glmnet)
library(lubridate)
library(e1071)
library(vtreat)
library(binaryLogic)
library(glmnet)
library(tidymodels)
library(Metrics)
library(Amelia)
library(naniar)
library(outliers)
library(ggpubr)
library(DAAG)
library(party)
library(rpart.plot)
library(pROC)
library(tree)
library(naniar)
library(Amelia)
library(mice)
library(rpart)
library(ROCR)
library(rattle)
library(adabag)
library(ipred) 
library(MLeval)

### Read in the train and test datasets

Train <- read.csv(file = 'Train1.csv')
Test <- read.csv(file = 'Test1.csv')
glimpse(Train)
glimpse(Test)

## First lets create the get modes function 

getmodes <- function(v,type=1) {
  tbl <- table(v)
  m1<-which.max(tbl)
  if (type==1) {
    return (names(m1)) #1st mode
  }
  else if (type==2) {
    return (names(which.max(tbl[-m1]))) #2nd mode
  }
  else if (type==-1) {
    return (names(which.min(tbl))) #least common mode
  }
  else {
    stop("Invalid type selected")
  }
}

getmodesCnt <- function(v,type=1) {
  tbl <- table(v)
  m1<-which.max(tbl)
  if (type==1) {
    return (max(tbl)) #1st mode freq
  }
  else if (type==2) {
    return (max(tbl[-m1])) #2nd mode freq
  }
  else if (type==-1) {
    return (min(tbl)) #least common freq
  }
  else {
    stop("Invalid type selected")
  }
}



############Data Preprocessing#####################################
##################################################################

# We will break the data down the train data into numeric and non-numeric variables to create a quality report

############### Numeric Data Processing ##############
Numeric_data <- Train %>% select_if(is.numeric)
Non_numeric_data <- Train %>% select_if(negate(is.numeric))


glimpse(Numeric_data)
glimpse(Non_numeric_data)

#From the data overview, admission type, admission source and discharge dispositions are categorical, labelled as 
# Which each numerical variable representing a specific factor category. These are nominal data
# We will does convert these into factor variables

Non_numeric_data$admission_type <-as.factor(Numeric_data$admission_type)
Non_numeric_data$discharge_disposition <- as.factor(Numeric_data$discharge_disposition)
Non_numeric_data$admission_source <- as.factor(Numeric_data$admission_source)
glimpse(Non_numeric_data)


#The converted columns are thus removed

Numeric_data <- Numeric_data[, -c(2,3,4)]
glimpse(Numeric_data)


# Data Quality Report Function
Q1 <- function(x, na.rm= TRUE) {
  quantile(x,na.rm=na.rm)[2]
}
Q3 <- function(x, na.rm= TRUE) {
  quantile(x,na.rm=na.rm)[4]
}

theNumericSummary <- function(x){
  c(length(x), n_distinct(x), sum(is.na(x)), mean(x,na.rm=TRUE),
    min(x,na.rm=TRUE), Q1(x,na.rm=TRUE), median(x,na.rm=TRUE),
    Q3(x,na.rm=TRUE), max(x,na.rm=TRUE), sd(x,na.rm=TRUE))
}
numericSummary <- Numeric_data %>% summarise(round(across(everything(),theNumericSummary)))
numericSummary <- cbind(stat = c("n", "unique", "missing", "mean", "min",
                                 "Q1", "median", "Q3", "max", "sd"), numericSummary)

numericSummaryFinal <- numericSummary %>%
  pivot_longer("patientID":"readmitted", names_to = "variable", values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(missing_pct = 100*missing/n, unique_pct = 100*unique/n) %>%
  dplyr::select(variable, n, missing, missing_pct, unique, unique_pct, everything())
#Final Data Quality Report Table
options(digits=3)
options(scipen=99)
numericSummaryFinal


# 11 out of the 14 numerical variables have no missing values 
# We see missing values in indicator level, time_in_hospital and number of lab procedures

# Lets visualize the missing values in the numeric data
gg_miss_var(Numeric_data)
missmap(Numeric_data, legend = TRUE)
view(miss_var_summary(Numeric_data))

#check the balance of the dataset using a bar plot of readmitted (target)
ggplot(data = Numeric_data, aes(x = readmitted, fill = as.factor(readmitted))) +
  geom_bar() + 
  scale_fill_manual(values = c("blue", "grey"), ylab('readmitted'))

#ggplot(data=Numeric_data, aes(x=readmitted, y=patientID, color=readmitted))+geom_histogram()

# We can also see that there is a fairly even distribution between the number of readmitted and non-readmitted patients 
#based on the patience ID. 


#################### Categorical Data Processing###########################
#create a new function called thefactorsummary
theFactorSummary <- function(x){
  c(length(x), n_distinct(x), sum(is.na(x)),
    getmodes(x, type=1), getmodesCnt(x, type=1),
    getmodes(x, type=-1), getmodesCnt(x, type=-1))
}
#apply function to all columns of datafactor and summarize 
factorSummary <- Non_numeric_data %>% summarise(across(everything(), theFactorSummary))

factorSummary <- cbind(stat = c("n", "unique", "missing", "1st mode",
                                "1st mode freq",
                                "least common", "least common freq"), factorSummary)

#create final data quality report for factor variables
factorSummaryFinal <- factorSummary %>%
  pivot_longer("race":"diabetesMed", names_to = "variable", values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(missing_pct = 100*as.double(missing)/as.double(n),
         unique_pct = 100*as.double(unique)/as.double(n)) %>%
  dplyr::select(variable, n, missing, missing_pct, unique, unique_pct, everything())

# Data Quality Report
options(digits=3)
options(scipen=99)
factorSummaryFinal

#Visualize Missing variables 
gg_miss_var(Non_numeric_data)
missmap(Non_numeric_data, legend = TRUE)
view(miss_var_summary(Non_numeric_data))

# We have missing values in medical specialty, payercode, race and diagnosis



#########################################################################
####MISSING VALUE IMPUTATIONS#########################################
heatmap(cor(Numeric_data, use= "complete.obs"))


##################### NUMERIC DATA IMPUTATIONS ########################
# We will use predictive mean matching 
# We see a good correlation between variables with missingness and num_medications

dfMiss <- Numeric_data # Create a new dataframe for housingNumeric
missing_1 <- is.na(dfMiss$indicator_level) # Get the missing data for LotFrontage
missing_2 <- is.na(dfMiss$time_in_hospital) # Get the missing data
missing_3 <- is.na(dfMiss$num_lab_procedures) # Get the missing data



#################### Populate dfMiss with missing_values indicators ########################
dfMiss$missing_1 <- missing_1 # Putting missing data in dfMiss
dfMiss$missing_2 <- missing_2 # Putting missing data in dfMiss
dfMiss$missing_3 <- missing_3 # Putting missing data in dfMiss

################## Predictive Mean Matching ############
#Imputation for indicator_level

# First create a new dataframe for plotting
#x<- Numeric_data$num_medications
#y<-Numeric_data$in
#z<-Numeric_data$number_diagnoses

#df<-data.frame(x,y,z)

dfMiss[missing_1,"indicator_level"] <- mice.impute.pmm(dfMiss$indicator_level, !dfMiss$missing_1, dfMiss$num_medications)
dfMiss[missing_2,"time_in_hospital"] <- mice.impute.pmm(dfMiss$time_in_hospital, !dfMiss$missing_2, dfMiss$num_medications)
dfMiss[missing_3,"num_lab_procedures"] <- mice.impute.pmm(dfMiss$num_lab_procedures, !dfMiss$missing_3, dfMiss$num_medications)


missmap(dfMiss, legend = TRUE) #Missing values have been imputed 
gg_miss_var(dfMiss)

numeric1=dfMiss

################ FACTOR VARIABLES IMPUTATIONS ############################
Non_numeric_data_1<-Non_numeric_data %>%
  mutate(across(everything(), ~replace_na(.x,getmodes(.x, type=1 ))))

missmap(Non_numeric_data_1, legend = TRUE)
gg_miss_var(Non_numeric_data_1) # Missing values in factor variables have been imputed 


########### FEATURE ENGINEERING AND SELECTION #####################
############# Numeric Data #############
heatmap(cor(dfMiss, use= "complete.obs"))

# We will select the variables that have good correlations with re-admitted
X_numeric= subset(dfMiss, select=c(patientID,number_inpatient,number_emergency, number_outpatient,num_medications,time_in_hospital,
                                   num_lab_procedures,number_diagnoses))

glimpse(X_numeric)

############## CATEGORICAL DATA ###################
## Select categorical variables based on domain knowledge and convert them to factor varibles
View(Non_numeric_data_1)

X_factor= subset(Non_numeric_data_1, select=c(race,gender, age,diagnosis,max_glu_serum,
                                              A1Cresult,metformin,repaglinide,nateglinide,chlorpropamide,
                                              glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,pioglitazone,
                                              rosiglitazone,acarbose,miglitol,troglitazone,
                                              insulin,glyburide-metformin))

X_factor_1 <- X_factor %>%
  mutate_if(sapply(X_factor, is.character), as.factor) #Converting character variables to factor variables 

glimpse(X_factor_1)

X_factor_1= subset(X_factor_1, select=-c(max_glu_serum.1
))
glimpse(X_factor_1)

traindata=cbind(X_numeric,X_factor_1)
View(traindata)
glimpse(traindata)


Y=dfMiss$readmitted
traindata=cbind(Y,traindata)
glimpse(traindata)
names(traindata)[1]='readmitted'
glimpse(traindata)

traindata$readmitted=as.factor(traindata$readmitted)
glimpse(traindata)

###################################################################################################

########## NUMERIC TRAIN DATA ##################
numeric_train=subset(traindata, select=c(readmitted,patientID,number_inpatient,number_emergency, number_outpatient,num_medications,time_in_hospital,
                                         num_lab_procedures,number_diagnoses) )

glimpse(numeric_train)

numeric_train_new= subset(numeric_train, select= -c(patientID))
glimpse(numeric_train_new)

#numeric_train_new$readmitted=as.integer(numeric_train_new$readmitted)
#glimpse(numeric_train_new)

################ NUMERIC TRAIN DATA WITH INTERACTIONS ######################



#####################################################################################################





############################## TEST DATA PREPARATION #################################################
Numeric_test <- Test %>% select_if(is.numeric)
Factor_test <- Test %>% select_if(negate(is.numeric))


glimpse(Numeric_test)
glimpse(Factor_test)

#From the data overview, admission type, admission source and discharge positions are categorical, labelled as 
# Which each numerical variable representing a specific factor category
# We will does convert these into factor variables

Factor_test$admission_type <-as.factor(Numeric_test$admission_type)
Factor_test$discharge_disposition <- as.factor(Numeric_test$discharge_disposition)
Factor_test$admission_source <- as.factor(Numeric_test$admission_source)
glimpse(Factor_test)


#The converted columns are thus removed

Numeric_test <- Numeric_test[, -c(2,3,4)]
glimpse(Numeric_test)


missmap(Numeric_test, legend = TRUE) #Missing values have been imputed 
gg_miss_var(Numeric_test)
view(miss_var_summary(Numeric_test))


#NUMERIC DATA IMPUTATIONS
# We will use predictive mean matching 
# We see a good correlation between variables with missingness and num_medications

dfMiss_a <- Numeric_test # Create a new dataframe for housingNumeric
missing_a <- is.na(dfMiss_a$indicator_level) # Get the missing data for LotFrontage
missing_b <- is.na(dfMiss_a$time_in_hospital) # Get the missing data



#Populate dfMiss with missing_values indicators
dfMiss_a$missing_a <- missing_a # Putting missing data in dfMiss
dfMiss_a$missing_b <- missing_b # Putting missing data in dfMiss

#Predict Mean Matching 
#Imputation for indicator_level

# First create a new dataframe for plotting
#x<- Numeric_data$num_medications
#y<-Numeric_data$in
#z<-Numeric_data$number_diagnoses

#df<-data.frame(x,y,z)

dfMiss_a[missing_a,"indicator_level"] <- mice.impute.pmm(dfMiss_a$indicator_level, !dfMiss_a$missing_a, dfMiss_a$num_medications)
dfMiss_a[missing_b,"time_in_hospital"] <- mice.impute.pmm(dfMiss_a$time_in_hospital, !dfMiss_a$missing_b, dfMiss_a$num_medications)


missmap(dfMiss_a, legend = TRUE) #Missing values have been imputed 
gg_miss_var(dfMiss_a)



#Factor Variables imputation 
Factor_test1<-Factor_test %>%
  mutate(across(everything(), ~replace_na(.x,getmodes(.x, type=1 ))))

Factor_test2 <- Factor_test1 %>%
  mutate_if(sapply(Factor_test1, is.character), as.factor)

missmap(Factor_test2, legend = TRUE)
gg_miss_var(Factor_test2) # Missing values in factor variables have been imputed

testdata=cbind(dfMiss_a,Factor_test2)

testdata1=subset(testdata, select=c(patientID,number_inpatient,number_emergency, number_outpatient,num_medications,time_in_hospital,
                                    num_lab_procedures,number_diagnoses,race,gender, age,diagnosis,max_glu_serum,
                                    A1Cresult,metformin,repaglinide,nateglinide,chlorpropamide,
                                    glimepiride,acetohexamide,glipizide,glyburide,tolbutamide,pioglitazone,
                                    rosiglitazone,acarbose,miglitol,troglitazone,
                                    insulin,glyburide-metformin))
glimpse(testdata1)

newtestdata=subset(testdata1, select=-c(number_outpatient.1))

glimpse(newtestdata)
glimpse(traindata)

patientID_test=newtestdata$patientID
Testdata_new=subset(newtestdata, select=-c(patientID))

glimpse(Testdata_new)

##########################################################################################################################

############################# NUMERIC TEST DATA ################################
test_numeric=subset(Testdata_new, select=c(number_inpatient,number_emergency, number_outpatient,num_medications,time_in_hospital,
                                           num_lab_procedures,number_diagnoses) )

glimpse(test_numeric)

################################################################
numeric_train_new= subset(numeric_train, select= -c(patientID))
glimpse(numeric_train_new)

#########################################################################################################################

#########################################################################################
############################# MODELLING #########################################
#######################################################################################

########## Basic Models IN Mars with Numeric data with and without interactions ###########
##Datasets
numeric_train_new
test_numeric
patientID_test

########### Without interactions
Marsmodel1<-earth(readmitted~.,data = numeric_train_new, degree=3,nk=50,pmethod="cv",nfold=10,ncross=5)
print(Marsmodel1)

Mpred=predict(Marsmodel1,test_numeric, type="response")
View(Mpred)

PredMPred<-replace(Mpred,Mpred>1,1)
myMARS <- data.frame(patientID = patientID_test, predReadmit= PredMPred)
write.csv(myMARS, file = 'myMARS_model.csv', row.names = F)

###### With interactions

Marsmodel<-earth(readmitted~num_medications*num_lab_procedures*number_inpatient, data = numeric_train_new, degree=3,nk=50,pmethod="cv",nfold=10,ncross=5)

predmars<-predict(Marsmodel,test_numeric, type="response")
View(predmars)
PredM<-replace(predmars,predmars>1,1)
sub_avg9 <- data.frame(patientID = patientID_test, predReadmit= PredM)
write.csv(sub_avg9, file = 'MARS_model_1.csv', row.names = F)




######################################################################################################
################################# RIGOROUS DATA PRE-PROCESSING ##############################################

####Training data
TrainOR=cbind(numeric1,Non_numeric_data_1)
glimpse(TrainOR)

TrainOR = subset(TrainOR, select=-c(missing_1,missing_2,missing_3))
Train_df=TrainOR

####Test data
testdata
testdata= subset(testdata, select=-c(missing_a,missing_b))
glimpse(testdata)


# Combine the train and test data
test_df=testdata
test_df$readmitted = NA
test_df$readmitted = as.factor(test_df$readmitted)
prediction_df <- rbind(Train_df,test_df)


glimpse(prediction_df)

# Check for missing values
gg_miss_var(prediction_df)
view(miss_var_summary(prediction_df)) # We only see missing values in the test re-admitted we created

#Convert medical specialty, payer code, race and diagnosis to factors
prediction_df$medical_specialty<-as.factor(prediction_df$medical_specialty)
prediction_df$payer_code<-as.factor(prediction_df$payer_code)
prediction_df$race<-as.factor(prediction_df$race)
prediction_df$diagnosis<-as.factor(prediction_df$diagnosis)

#Checking the relationship between the variables and admitted

value <- abs(rnorm(57855 , 0 , 15))

ggplot(prediction_df[1:57855,], aes(fill=readmitted, y=value, x=race)) + 
  geom_bar(position="stack", stat="identity")

ggplot(prediction_df[1:57855,], aes(fill=readmitted, y=value, x=medical_specialty)) + 
  geom_bar(position="stack", stat="identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(prediction_df[1:57855,], aes(fill=readmitted, y=value, x=age)) + 
  geom_bar(position="stack", stat="identity")

#We will drop some columns which do not affect the target variable

drop <- c("examide", "citoglipton", "discharge_disposition", "race")
prediction_df = prediction_df[,!(names(prediction_df) %in% drop)]

#######Encoding #####

prediction_df$gender <- as.numeric(as.factor(prediction_df$gender))
prediction_df$age <- as.numeric(as.factor(prediction_df$age))
prediction_df$payer_code <- as.numeric(prediction_df$payer_code)
prediction_df$medical_specialty <- as.numeric(prediction_df$medical_specialty)
prediction_df$diagnosis <- as.numeric(prediction_df$diagnosis)
#prediction_df$readmitted <- as.numeric(as.factor(prediction_df$readmitted))
prediction_df$insulin <- as.numeric(as.factor(prediction_df$insulin))
prediction_df$diabetesMed <- as.numeric(as.factor(prediction_df$diabetesMed))
prediction_df$metformin <- as.numeric(as.factor(prediction_df$metformin))

glimpse(prediction_df)

# Max Glu Serum
prediction_df$max_glu_serum <- as.numeric(as.factor(prediction_df$max_glu_serum))
prediction_df$max_glu_serum[prediction_df$max_glu_serum == 1] <- 1
prediction_df$max_glu_serum[prediction_df$max_glu_serum == 2] <- 1
prediction_df$max_glu_serum[prediction_df$max_glu_serum == 4] <- 1
prediction_df$max_glu_serum[prediction_df$max_glu_serum == 3] <- 0

# A1 Cresult
prediction_df$A1Cresult <- as.numeric(as.factor(prediction_df$A1Cresult))
prediction_df$A1Cresult[prediction_df$A1Cresult == 1] <- 1
prediction_df$A1Cresult[prediction_df$A1Cresult == 2] <- 1
prediction_df$A1Cresult[prediction_df$A1Cresult == 4] <- 1
prediction_df$A1Cresult[prediction_df$A1Cresult == 3] <- 0

# Repaglinide
prediction_df$repaglinide <- as.numeric(as.factor(prediction_df$repaglinide))
prediction_df$repaglinide[prediction_df$repaglinide == 1] <- 1
prediction_df$repaglinide[prediction_df$repaglinide == 3] <- 1
prediction_df$repaglinide[prediction_df$repaglinide == 4] <- 1
prediction_df$repaglinide[prediction_df$repaglinide == 2] <- 0

# Chlorpropamide
prediction_df$chlorpropamide <- as.numeric(as.factor(prediction_df$chlorpropamide))
prediction_df$chlorpropamide[prediction_df$chlorpropamide == 2] <- 1
prediction_df$chlorpropamide[prediction_df$chlorpropamide == 3] <- 1
prediction_df$chlorpropamide[prediction_df$chlorpropamide == 4] <- 1
prediction_df$chlorpropamide[prediction_df$chlorpropamide == 1] <- 0

# Glimepiride
prediction_df$glimepiride <- as.numeric(as.factor(prediction_df$glimepiride))
prediction_df$glimepiride[prediction_df$glimepiride == 1] <- 1
prediction_df$glimepiride[prediction_df$glimepiride == 3] <- 1
prediction_df$glimepiride[prediction_df$glimepiride == 4] <- 1
prediction_df$glimepiride[prediction_df$glimepiride == 2] <- 0

# Glipizide
prediction_df$glipizide <- as.numeric(as.factor(prediction_df$glipizide))
prediction_df$glipizide[prediction_df$glipizide == 1] <- 1
prediction_df$glipizide[prediction_df$glipizide == 3] <- 1
prediction_df$glipizide[prediction_df$glipizide == 4] <- 1
prediction_df$glipizide[prediction_df$glipizide == 2] <- 0

# Glyburide
prediction_df$glyburide <- as.numeric(as.factor(prediction_df$glyburide))
prediction_df$glyburide[prediction_df$glyburide == 1] <- 1
prediction_df$glyburide[prediction_df$glyburide == 3] <- 1
prediction_df$glyburide[prediction_df$glyburide == 4] <- 1
prediction_df$glyburide[prediction_df$glyburide == 2] <- 0

# Pioglitazone
prediction_df$pioglitazone <- as.numeric(as.factor(prediction_df$pioglitazone))
prediction_df$pioglitazone[prediction_df$pioglitazone == 1] <- 1
prediction_df$pioglitazone[prediction_df$pioglitazone == 3] <- 1
prediction_df$pioglitazone[prediction_df$pioglitazone == 4] <- 1
prediction_df$pioglitazone[prediction_df$pioglitazone == 2] <- 0

# Rosiglitazone
prediction_df$rosiglitazone <- as.numeric(as.factor(prediction_df$rosiglitazone))
prediction_df$rosiglitazone[prediction_df$rosiglitazone == 1] <- 1
prediction_df$rosiglitazone[prediction_df$rosiglitazone == 3] <- 1
prediction_df$rosiglitazone[prediction_df$rosiglitazone == 4] <- 1
prediction_df$rosiglitazone[prediction_df$rosiglitazone == 2] <- 0

glimpse (prediction_df)

# Select appropriate columns

prediction_df = prediction_df[,c(1:20,22,23,25,26,28,29,34,40,41)] 

#Separate the train and test data 
train_df_1 <- prediction_df[1:57855,]
test_df_1 <- prediction_df[57856:96423,]

glimpse(train_df_1)
train_readmmit=train_df_1$readmitted

test_ID=test_df_1$patientID
Train_df_new = subset(train_df_1, select=-c(patientID))
Test_df_new = subset(test_df_1, select=-c(patientID))
Test_df_new = subset(Test_df_new, select=-c(readmitted))
glimpse(Train_df_new)
glimpse(Test_df_new)

Train_df_new$readmitted=as.factor(Train_df_new$readmitted)

glimpse(Train_df_new)




#############################  MODELLING ##########################################################

###########################################################################################################
########################## Random Forest ##################################################################

set.seed(20)
control <- trainControl(method="repeatedcv", number=5, search="grid")
tunegrid <- expand.grid(.mtry=c(1:25, by=5))
rf_CV = train(readmitted~., data=Train_df_new, method="rf", tunegrid=tunegrid, trControl=control)
print(rf_CV)

summary(rf_CV)
attributes(rf_CV)

varImp(rf_CV)

submission_rf <- data.frame(test_ID, pred_rf_2)
###################################################################################
##### Actual Predictions #########################################################

ref_pred_train=predict(rf_CV,Train_df_new)

##########################################################################
########### Probabilistic predictions #########################################
pred_rf_2 = predict(rf_CV,Test_df_new,type="prob")

pred_rf_2

submission_rf <- data.frame(test_ID, pred_rf_2)
write_csv(submission_rf, "submission_rf2.csv")

########### Confusion matrix #######################
train_readmmit=as.factor(train_readmmit)
glimpse(train_readmmit)

confusionMatrix(ref_pred_train,train_readmmit,
                mode = "everything",
                positive="1")
######################################################




################## GRADIENT BOOSTING##################################
control_boost <- trainControl(method="repeatedcv", number=10, search="grid")

set.seed(123)
model_xgboost <- train(
  readmitted ~., data = Train_df_new, method = "xgbTree",
  trControl = control_boost, metric="Accuracy")
# Best tuning parameter
model_final_xgboost=model_xgboost$bestTune

print(model_final_xgboost)

model_xgboost

varImp(model_xgboost)

print(model_xgboost)
summary(model_xgboost)


####Predictions######
xgboost_pred= predict(model_xgboost, Test_df_new, type="prob")


xgboost_final <- data.frame(test_ID, xgboost_pred)
write_csv(xgboost_final, "submission_xgboost.csv")



####Confusion Matrix###

train_readmmit=as.factor(train_readmmit)
train_pred_xg= predict(model_xgboost, Train_df_new)
train_pred_xg

set.seed(19)
confusionMatrix(train_pred_xg,train_readmmit,
                mode = "everything",
                positive="1")

########### Log Loss############
LogLoss=function(actual, predicted)
{
  result=-1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}

LogLoss(train_readmmit_num, train_pred_xg_num)
########## Lift Curve ##########

pred2 <- prediction(predictionRF,TrainPF1$Target)    #RCumulative Lift curve for training data
perf2 <- performance(pred2,"lift","rpp")
plot(perf2,main="lift curve",colorize=T, cumulative=T)


######ROC Curve ###############
train_pred_xg_l=as.numeric(train_pred_xg)
train_readmmit_l= as.numeric(train_readmmit)

PRED <- prediction(train_pred_xg_l,train_readmmit_l)    #ROC curve for training data
PERF1 <- performance(PRED,"tpr","fpr") 

dev.off()
plot(PERF1,colorize=TRUE, print.cutoffs.at = c(0.25,0.5,0.75)); #Plotting of ROC
abline(0, 1, col="red") 

#Accuracy by average cutoff
PERF1 <- performance(PRED, "acc")
plot(PERF1, avg= "vertical",  
     spread.estimate="boxplot", 
     show.spread.at= seq(0.1, 0.9, by=0.1))

############

#install.packages('verifications')
#library(verifications)
#roc.plot(model_xgboost$pred,model_xgboost$pred)


################## Insights ##################



#######################################################################



##################################################################################################
############# Logistic Regression #######################################################################
set.seed(10)
lm2 <- glm(readmitted~.,data=Train_df_new, family = binomial)
#lm2 <- glm(readmitted~.,data=Train_df_new, family = binomial, metric="Accuracy")
summary(lm2)
print(lm2)


######################### re-tune logistic regression model ######################
set.seed(400)
control <- trainControl(method="cv", number=5)
lm5= train(readmitted~., data=Train_df_new, method="glm", metric="Accuracy", trControl=control)
print(lm5)

############### re-tune logistic model #########
set.seed(100)
control_2 <- trainControl(method="repeatedcv", number=10, search="grid")

lm_final= train(readmitted~., data=Train_df_new, method="glm", metric="Accuracy", trControl=control_2)
print(lm_final)
summary(lm_final)
########### Predictions #####################

pred_lm_2 = predict(lm2, Test_df_new, type="response")
pred_lm_5 = predict(lm5, Test_df_new, type="prob")
pred_lm_final = predict(lm_final, Test_df_new, type="prob")
pred_lm_final

Read_2= predict(lm5, Test_df_new, type="prob")
Read_2
library(xlsx)

Logistic_final <- data.frame(test_ID, pred_lm_final)
write_csv(Logistic_final, "submission_lg_final.csv")

###########################################################################################
######## KNN ########################################################################################
set.seed(67)
knnControl <- trainControl(method="repeatedcv", number=10) 

knn <- train(readmitted~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = knnControl,
             metric     = "Accuracy",
             data       = numeric_train_new, prob=TRUE)
print(knn)

knn$bestTune
attributes(knn)

predknn = predict(knn,Test_df_new, type="prob")



View(predknn)
submissionknn <- data.frame(test_ID, predknn)
write_csv(submissionknn, "submissionknn")

######################################################################################################
#### svm #############################################################################################
set.seed(148)
SVM<- ksvm(readmitted ~.,data= Train_df_new,prob.model=TRUE, metric="Accuracy", trControl=control_2)
#SVM_1<- ksvm(readmitted ~.,data= Train_df_new,prob.model=TRUE, metric="Accuracy") 
print(SVM)
summary(SVM)
predSVM<- predict(SVM, Test_df_new,type="prob")

sub_SVM <- data.frame(patientID = test_ID, predReadmit= predSVM)
write.csv(sub_SVM, file = 'SVM_final.csv', row.names = F)


######Confusion Matrix######

train_readmmit=as.factor(train_readmmit)
train_pred_svm= predict(SVM, Train_df_new)

confusionMatrix(train_pred_svm,train_readmmit,
                mode = "everything",
                positive="1")
