library(ranger)
library(randomForestSRC)
library(Rborist)
library(h2o)
library(xgboost)
library(lightgbm)

library(MetricsWeighted)
library(splitTools)

set.seed(1)
n <- 10000
x1 <- seq_len(n)
x2 <- rnorm(n)
x3 <- rexp(n)
x4 <- runif(n)
X <- cbind(x1, x2, x3, x4)
y <- rnorm(n, x1 / 1000 + x2 / 10 + x3 / 5)

idx <- partition(y, p = c(train = 0.7, valid = 0.3))
X_train <- X[idx$train, ]
y_train <- y[idx$train]
X_valid <- X[idx$valid, ]
y_valid <- y[idx$valid]

mtry <- floor(sqrt(ncol(X)))
mtry_p <- mtry / ncol(X)

# XGB Random Forest

system.time({
  param_xgb <- list(max_depth = 10,
                    learning_rate = 1,
                    objective = "reg:linear",
                    subsample = 0.63,
                    lambda = 0,
                    alpha = 0,
                    min_child_weight = 5,
                    # tree_method = "hist",
                    # base_score
                    # max_delta_step
                    colsample_bynode = mtry_p)
  
  dtrain_xgb <- xgb.DMatrix(X_train, label = y_train)
  
  fit_xgb <- xgb.train(param_xgb,
                       dtrain_xgb,
                       nrounds = 1,
                       num_parallel_tree = 500)
  pred_xgb <- predict(fit_xgb, X_valid)
})


# LGB Random Forest

system.time({
  param_lgb <- list(boosting = "rf",
                    max_depth = 10,
                    num_leaves = 1000,
                    learning_rate = 1,
                    objective = "regression",
                    bagging_fraction = 0.63,
                    bagging_freq = 1,
                    reg_lambda = 0,
                    reg_alpha = 0,
                    min_data_in_leaf = 5,
                    colsample_bynode = mtry_p)
  
  dtrain_lgb <- lgb.Dataset(X_train, label = y_train)
  
  fit_lgb <- lgb.train(param_lgb,
                       dtrain_lgb, nthread=8,
                       verbose = -1,
                       nrounds = 500)
  
  pred_lgb <- predict(fit_lgb, X_valid)
})

# Ranger
system.time({
  fit_ranger <- ranger(y = y_train, 
                       x = X_train, 
                       mtry = mtry,
                       min.node.size = 5,
                       max.depth = 10, 
                       num.trees = 500)
  
  pred_ranger <- predict(fit_rf, X_valid)$predictions
})

# Rborist
system.time({
  fit_rbor <- Rborist(x = X_train, 
                      y = y_train, 
                      nTree = 500, 
                      autoCompress = mtry_p,  
                      minInfo = 0,
                      nLevel = 10,
                      minNode = 5)
  pred_rbor <- predict(fit_rbor, X_valid)$yPred
})

system.time({
  fit_rfsrc <- rfsrc(reformulate(colnames(X_train), "y_train"), 
                     data = data.frame(X_train, y_train), 
                     mtry = mtry,
                     ntree = 500, 
                     nodedepth = 10,
                     nodesize = 5,
                     seed = 837363)
  pred_rfsrc <- predict(fit_rfsrc, data.frame(X_valid))$predicted
})

# Evaluate predicitons
pred <- data.frame(
  pred_xgb,
  pred_lgb,
  pred_ranger,
  pred_rbor,
  pred_rfsrc
)

summary(pred) 
cor(pred)

rmse <- function(pred, y) {
  sqrt(mean((y - pred)^2))
}

sapply(pred, rmse, y_valid)

sessionInfo()
