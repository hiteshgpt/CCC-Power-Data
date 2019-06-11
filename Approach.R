# Load required libraries #
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")

#########################
#Load csv data in R
#########################

dl <- tempfile()
download.file("https://raw.githubusercontent.com/hiteshgpt/CCC-Power-Data/master/Folds5x2_pp.csv", dl)
dataset <- read.csv(dl)
rm(dl)

# Validate the structure of the dataset
str(dataset)

#########################
# Partion the data in test and training sets (50%) each
#########################

set.seed(1)
test_index<-createDataPartition(dataset$PE,times=1,p=0.5,list=FALSE)
test_set<-dataset[test_index,]
train_set<-dataset[-test_index,]

# Create a mini sample of 4% of our training data set to plot 
# graphs with few data points
set.seed(1)
mini_index<-createDataPartition(train_set$PE,times=1,p=0.04,list=FALSE)
mini_set<-dataset[mini_index,]

#########################
#Data Inferences
#########################

# Scatter plot for each input variable to validate linear correlation
train_set %>% 
  ggplot(aes(AT, PE)) + 
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm")

train_set %>% 
  ggplot(aes(V, PE)) + 
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm")

train_set %>% 
  ggplot(aes(AP, PE)) + 
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm")

train_set %>% 
  ggplot(aes(RH, PE)) + 
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm")

# Check for Bi-variate normal distribution for RH
train_set %>%
  mutate(z_RH = round((RH - mean(RH)) / sd(RH))) %>%
  filter(z_RH %in% -5:5) %>%
  ggplot() +  
  stat_qq(aes(sample = PE)) +
  facet_wrap( ~ z_RH) 

# Check for confounding variables by finding correlation
train_set %>%  
  summarize(cor(AT, AP), cor(AP, RH), cor(AT,RH))

# Stratify AT and plot AP vs PE to check causality
train_set %>% mutate(AT_strata = round_any(AT, 5)) %>%
                       filter(AT >= 5 & AT <=30)%>%
  ggplot(aes(AP, PE)) +  
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  facet_wrap( ~ AT_strata)

# Stratify AT and plot RH vs PE to check causality
train_set %>% mutate(AT_strata = round_any(AT, 5)) %>%
  filter(AT >= 5 & AT <=30)%>%
  ggplot(aes(RH, PE)) +  
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  facet_wrap( ~ AT_strata)

#########################
#Model Evaluations
#########################

# Create an LSE function to repeatedly check the LhSE values for our various models

LSE <- function(actual, predicted){
  mean((actual - predicted)^2)
}

#########################
# Model 1: Average PE of Training Data
#########################
y_hat <- mean(train_set$PE)
LSE_average<- LSE(test_set$PE, y_hat)
paste("LSE - Average =",LSE_average,sep = " ")

# Add the output to a results table. We will use this table 
# to compare various models

LSE_table <- data_frame(method = "Random Guess", 
                         LSE = LSE_average)

#########################
# Model 2: Use Linear Regression
#########################

fit <- lm(PE ~ ., data = train_set)
y_hat <- predict(fit, test_set)
LSE_lm <- LSE(test_set$PE,y_hat)
paste("LSE - Linear Regression =",LSE_lm, sep = " ")

# Check Summary metric for this
summary(fit)

# Add the output to a results table
LSE_table <- bind_rows(LSE_table, data_frame(method="Linear Regression",  
                                               LSE = LSE_lm))

#########################
# Model 3: Use K Nearest Neighbours
#########################

knn_fit<-train(PE ~ .,method="knn", data=train_set,
               tuneGrid = data.frame(k = seq(1, 15, 2))) 
y_hat_knn <- predict(knn_fit, test_set, type="raw")
LSE_knn <- LSE(test_set$PE,y_hat_knn)
paste("LSE - KNN =",LSE_knn, sep = " ")

# Add the output to a results table
LSE_table <- bind_rows(LSE_table, data_frame(method="KNN", 
                                             LSE = LSE_knn))

# To find Cross-validation values of K
ggplot(knn_fit, highlight = TRUE)

# The best value of K is found using the code below
knn_fit$bestTune

#Plot the predicted output on our sample mini set for AT var
#to see visually see the results. 
mini_fit<-train(PE ~ .,method="knn", data=mini_set, 
                tuneGrid = data.frame(k = seq(1, 15, 2)))
mini_set %>% 
  mutate(y_hat = predict(mini_fit)) %>% 
  ggplot() +
  geom_point(aes(AT, PE)) +
  geom_step(aes(AT, y_hat), col="red")

#########################
# Model 4: Use Regression Trees
#########################

rpart_fit<-train(PE ~ .,method="rpart", data=train_set, 
                 tuneGrid = data.frame(cp = seq(0, 0.5, len = 25))) 
y_hat_rpart <- predict(rpart_fit, test_set)
LSE_rpart <- LSE(test_set$PE,y_hat_rpart)
paste("LSE - Rpart =",LSE_rpart, sep = " ")

# Add the output to a results table
LSE_table <- bind_rows(LSE_table, data_frame(method="Regression Tree", 
                                             LSE = LSE_rpart))

# To find the Cross Validation of CP
ggplot(rpart_fit)

#Plot the predicted output on our sample mini set for AT var
#to see visually see the results. 

mini_fit<-train(PE ~ .,method="rpart", data=mini_set, 
                tuneGrid = data.frame(cp = seq(0, 0.5, len = 25)))
mini_set %>% 
  mutate(y_hat = predict(mini_fit)) %>% 
  ggplot() +
  geom_point(aes(AT, PE)) +
  geom_step(aes(AT, y_hat), col="red")

plot(mini_fit$finalModel, margin = 0.1)
text(mini_fit$finalModel, cex = 0.75)

#########################
# Model 4: Use Random Forest
#########################

rf_fit<-train(PE ~ .,method="Rborist", data=train_set) 
y_hat_rf <- predict(rf_fit, test_set)
LSE_rf <- LSE(test_set$PE,y_hat_rf)
paste("LSE - Random Forest =",LSE_rf, sep = " ")

# Add the output to a results table
LSE_table <- bind_rows(LSE_table, data_frame(method="Random Forest", 
                                             LSE = LSE_rf))

# To find the Error Rate as a function of number of trees
plot(rf_fit)

#Plot the predicted output on our sample mini set for AT var
#to see visually see the results. 

mini_fit<-train(PE ~ .,method="Rborist", data=mini_set)
mini_set %>% 
  mutate(y_hat = predict(mini_fit)) %>% 
  ggplot() +
  geom_point(aes(AT, PE)) +
  geom_step(aes(AT, y_hat), col="red")
