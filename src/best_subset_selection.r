
# 'leaps' library for best subset selection
library(leaps)
library(dplyr)
library(tidyr)

# set the directory to where dataset is
setwd(dir = '~/Desktop/Regression Analysis/')

# read ISTANBUL STOCK EXCHANGE dataset
ise_df = read.csv(file = 'data/istanbul_stock_exchange.csv', sep = ',')

# split data into train and test set
# Year 2009, 2010 records will be part of training dataset and 
# records of year 2011 will be included in test set
# https://rpubs.com/bradleyboehmke/data_wrangling
# https://tidyr.tidyverse.org/reference/separate.html
ise_df = ise_df %>% separate(date, c('day', 'month', 'year'), remove = TRUE)
ise_df_train = subset(ise_df, year < 11)
ise_df_test = subset(ise_df, year == 11)

# extract relevant columns for regression
ise_df_train = ise_df_train[, c('SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM','USD_BASED_ISE')]
ise_df_test = ise_df_test[, c('SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM','USD_BASED_ISE')]

# training and test set shape
print(dim(ise_df_train))
print(dim(ise_df_test))

# ======================================
# ALGORITHM-1 : BEST SUBSET SELECTION
# ======================================
# use 'regsubsets' function from 'leaps' package to perform best subset selection
# Note that 'regsubsets' uses 'RSS' as a criterion for 'best'
# 'nvmax' here is set to 7 so that all 7 predictor variables are used for subset selection
# 'nbest' is set to 1 so that 1 best model is selected when regression model
# uses num. of predictors = 1, 2,...7
best_subset = regsubsets(USD_BASED_ISE ~ ., 
                         data = ise_df_train, 
                         nvmax = 7, 
                         nbest = 1,
                         method = 'exhaustive')
best_subset_summary = summary(best_subset)

# make a table of R^2, adj. R^2, Mallows' Cp statistics for above identified best models
# so that best of the best model can be identified by analyzing the table as described in 
# https://onlinecourses.science.psu.edu/stat501/node/330/
num_predictors = c(1,2,3,4,5,6,7)
RSS = best_subset_summary$rss
R_sqr = best_subset_summary$rsq
Adj_R_sqr = best_subset_summary$adjr2
Mallows_Cp = best_subset_summary$cp
best_subset_summary_table = data.frame(num_predictors,
                                       RSS, 
                                       R_sqr, 
                                       Adj_R_sqr, 
                                       Mallows_Cp)

# =========================================
# ALGORITHM-2 : FORWARD STEPWISE SELECTION
# =========================================
# use 'regsubsets' function from 'leaps' package to perform forward subset selection
# Note that 'regsubsets' uses 'RSS' as a criterion for 'best'
forward_stepwise = regsubsets(USD_BASED_ISE ~ .,
                              data = ise_df_train,
                              nvmax = 7, 
                              method = 'forward')
forward_stepwise_summary = summary(forward_stepwise)

# create a table consisting of statistics R^2, adj. R^2, Mallows' Cp
num_predictors = c(1,2,3,4,5,6,7)
RSS = forward_stepwise_summary$rss
R_sqr = forward_stepwise_summary$rsq
Adj_R_sqr = forward_stepwise_summary$adjr2
Mallows_Cp = forward_stepwise_summary$cp
forward_stepwise_summary_table = data.frame(num_predictors, 
                                            RSS, 
                                            R_sqr, 
                                            Adj_R_sqr, 
                                            Mallows_Cp)

# =========================================
# ALGORITHM-3 : BACKWARD STEPWISE SELECTION
# =========================================
# use 'regsubsets' function from 'leaps' package to perform backward stepwise selection
# Note that 'regsubsets' uses 'RSS' as a criterion for 'best'
backward_stepwise = regsubsets(USD_BASED_ISE ~ .,
                               data = ise_df_train,
                               nvmax = 7, 
                               method = 'backward')
backward_stepwise_summary = summary(backward_stepwise)

# create a table consisting of statistics R^2, adj. R^2, Mallows' Cp
num_predictors = c(1,2,3,4,5,6,7)
RSS = backward_stepwise_summary$rss
R_sqr = backward_stepwise_summary$rsq
Adj_R_sqr = backward_stepwise_summary$adjr2
Mallows_Cp = backward_stepwise_summary$cp
backward_stepwise_summary_table = data.frame(num_predictors, 
                                             RSS, 
                                             R_sqr, 
                                             Adj_R_sqr, 
                                             Mallows_Cp)

# Although scores from all 3 tables look identical, let's double check 
# if these algorithms select different models by any chance
print(coef(best_subset, 5))
print(coef(forward_stepwise, 5))
print(coef(backward_stepwise, 5))

# FORWARD and BACKWARD STEPWISE SELECTION along with BEST SUBSET SELECTION are selecting same models 
# as best model. So it is not relevant to implement and/or analyze FORWARD and BACKWARD methods

# =========================================
#             MODEL SELECTION
# =========================================
# There are two ways to select our final model

# 1. Eyeball R^2, adj. R^2, Mallows' Cp statistics and make a selection -- Statisticians do this
# This strategy is best decsribed in https://onlinecourses.science.psu.edu/stat501/node/330/

# 2. Implement Cross-Validation -- Computer Scientists do this

# =========================================
#             CROSS-VALIDATION
# =========================================
# 10-fold Cross-Validation will be implemented here with best subset selection. 
# We will generate 10 folds of training and validation sets. Across each fold, we will perform
# best subset selection. In the end, a 10x7 matrix of validation error scores (predictions on validation set)
# will be generated. This matrix is averaged column wise to get average validation scores
# across models with 1 predictor, 2 predictors ..... 7 predictors.
# Column with least mean validation error score is chosen for our final model. Note that selecting
# least mean validation error score only gives us the number of predictors that need to be in our
# final model but not the model itself. Once we have the count of predictors that need to 
# go into our final model, we perform best subset selection across models with that many 
# number of predictors

k = 10
set.seed(12345)
val_errors_matrix = matrix(NA, k, 7, dimnames = list(NULL, paste(1:7)))

# generate data folds for validation process
folds = sample(1:k, 
               nrow(ise_df_train),
               replace = TRUE)

# outer loop iterated across folds
for(fold in 1:k)
{
  # create training and validation data
  train_data = ise_df_train[folds != fold, ]
  validation_data = ise_df_train[folds == fold, ]

  # perform best subset selection (training) on corresponding fold -- TRAIN phase
  # This step outputs best model for 1,2.....k predictors
  best_fit = regsubsets(USD_BASED_ISE ~ .,
                        data = train_data,
                        nvmax = 7)
  
  # VALIDATION phase
  # Use best model identified from above and evaluate CV score
  for(nPred in 1:7)
  {
    # 'regsubsets' doesn't have 'predict' method
    # we need to write our own prediction and error calculation steps
    betas = coef(best_fit, nPred)

    # create 'X_valid' by adding column of 1s. Let's call the column '(Intercept)' for future use
    X_valid = validation_data[,c('SP','DAX','FTSE','NIKKEI','BOVESPA','EU','EM')]
    X_valid['(Intercept)'] = 1
    X_valid = X_valid[,c('(Intercept)', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM')]
    X_valid = X_valid[,c(names(betas))]

    # create 'y_test'
    y_valid = validation_data$USD_BASED_ISE

    # perform (beta * X) matrix multiplication
    y_pred = as.matrix(X_valid) %*% as.matrix(betas)
    
    # add validation error scores (MSE) to error matrix 
    val_errors_matrix[fold, nPred] = mean((y_valid-y_pred)^2)
  }  
}

# avg. validation errors column-wise and choose the index with min. avg value
mean_val_errors = apply(val_errors_matrix, 2, mean) 
plot(mean_val_errors, type = 'b', xlab = 'Number of predictors', ylab = 'Mean Error')

# Looks like number of predictors = 3 has lowest validation error score
# Select best model among all possible combinations of 3 predictor variables
best_model = regsubsets(USD_BASED_ISE ~ ., data = ise_df_train, nvmax = 7)
best_model_coef = coef(best_model, 3)
print(best_model_coef)

# THE 3 PRECICTOR VARIABLE MODEL OUTPUT BY BEST-SUBSET SELECTION CONTAINS PREDICTORS 'BOVESPA', 'EU', 'EM'

# Prediction on test set
test = ise_df_test[ , c('BOVESPA', 'EU', 'EM')]
test['(Intercept)'] = 1
test = test[ , c('(Intercept)', 'BOVESPA', 'EU', 'EM')]

y_test = ise_df_test$USD_BASED_ISE
y_pred = as.matrix(test) %*% best_model_coef

# R^2 score
test_rss = sum((y_test - y_pred)^2)
test_tss = sum((y_test - mean(y_test))^2)
test_r2 = 1 - (test_rss/test_tss)
print(test_r2)  

# adjusted R^2 score
numerator = (1-test_r2) * (dim(ise_df_test)[1] - 1)
denominator = (dim(ise_df_test)[1] - 3 - 1)
test_adj_r2 = 1 - (numerator/denominator)
print(test_adj_r2)


