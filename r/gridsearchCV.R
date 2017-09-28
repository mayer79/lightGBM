library(ggplot2) # for diamonds data
library(lightgbm)
library(glmnet)

head(diamonds)

# data preparation
diamonds <- transform(as.data.frame(diamonds),
                      log_price = log(price),
                      log_carat = log(carat),
                      cut = as.numeric(cut),
                      color = as.numeric(color),
                      clarity = as.numeric(clarity))
summary(diamonds)

# Input used
x <- c("log_carat", "cut", "color", "clarity", "depth", "table")

# Train/test split
set.seed(3928272)
.in <- sample(c(FALSE, TRUE), 
              nrow(diamonds), 
              replace = TRUE, 
              p = c(0.15, 0.85))

train <- list(X = as.matrix(diamonds[.in, x]),
              y = as.numeric(diamonds[.in, "log_price"]))

test <- list(X = as.matrix(diamonds[!.in, x]),
             y = as.numeric(diamonds[!.in, "log_price"]))

#======================================================================
# Function to calculate root mean squared error
#======================================================================

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}


#======================================================================
# Multiple linear regression for comparison
#======================================================================

fit_lm <- glmnet(train$X, y = train$y, lambda = 0)
rmse(test$y, predict(fit_lm, test$X)) # 0.1455686

#======================================================================
# Toy example with lightgbm (no tuning)
#======================================================================

dtrain <- lgb.Dataset(train$X, 
                      label = train$y)

params <- list(learning_rate = 0.1)

system.time(fit_lgb <- lgb.train(params = params,
                                 data = dtrain,
                                 nrounds = 200,
                                 objective = "regression",
                                 verbose = 0L)) # 0.8 seconds

pred <- predict(fit_lgb, test$X)
rmse(test$log_price, pred) # 0.09566155

#======================================================================
# Same example but using cross-validation for determining nrounds
#======================================================================

system.time(fit_lgb2 <- lgb.cv(params = params,
                               data = dtrain,
                               nrounds = 1000, # we use early stopping anyway
                               nfold = 5,
                               objective = "regression",
                               eval = "rmse",
                               showsd = FALSE,
                               early_stopping_rounds = 5,
                               verbose = 0L))
info <- as.list(fit_lgb2)
info$best_iter # 285
-info$best_score # 0.0971542 (rmse)

# so running around 285 boosts will be enough for that particular parameter set


#======================================================================
# Tuned by gridsearch CV
#======================================================================

# tuned with cross-validation grid-search (takes 30 minutes for full search)

paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.1, 0.05, 0.01),
                         num_leaves = 2^(5:7) - 1,
                         min_data_in_leaf = c(20, 40),
                         feature_fraction = c(0.8, 1),
                         bagging_fraction = c(0.6, 0.8, 1),
                         bagging_freq = 4,
                         nthread = 4)

(n <- nrow(paramGrid)) # 108

for (i in seq_len(n)) {
  print(i)
  gc(verbose = FALSE) # clean memory

  cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
                dtrain,     
                nrounds = 1000, # we use early stopping
                nfold = 5,
                objective = "regression",
                showsd = FALSE,
                early_stopping_rounds = 5,
                verbose = 0L)
  
  paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  save(paramGrid, file = "paramGrid.RData") # if lgb crashes
}

# load("paramGrid.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ])

# Use best m
m <- 5

# keep test predictions, no model
predList <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  print(i)
  gc(verbose = FALSE) # clean memory
  
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = dtrain, 
                        nrounds = paramGrid[i, "iteration"],
                        objective = "regression",
                        verbose = 0L)
  
  predList[[i]] <- predict(fit_temp, test$X)
}

pred <- rowMeans(do.call(cbind, predList))
rmse(test$y, pred) # 0.09437292
