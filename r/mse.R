library(ranger)
library(randomForestSRC)
library(h2o)
library(xgboost)
library(lightgbm)

library(MetricsWeighted)
library(splitTools)

source("r/functions.R")

h2o.init()

# Create data
head(X <- make_X())
head(y <- make_y(X, "mse", strength = 5))
mean(y)
hist(y)

mtry <- floor(sqrt(ncol(X)))
mtry_p <- mtry / ncol(X)

# Split into training and validation
idx <- partition(y, p = c(train = 0.7, valid = 0.3))
X_train <- X[idx$train, ]
y_train <- y[idx$train]
X_valid <- X[idx$valid, ]
y_valid <- y[idx$valid]

train <- data.frame(y_train, X_train)
valid <- data.frame(y_valid, X_valid)

train_h2o <- as.h2o(train)
valid_h2o <- as.h2o(valid)

# Ranger
system.time({ # 0
  fit_ranger <- ranger(y = y_train, 
                       x = X_train, 
                       mtry = mtry,
                       min.node.size = 5,
                       max.depth = 10, 
                       num.trees = 500)
  
  pred_ranger <- predict(fit_ranger, X_valid)$predictions
})

system.time({ # 4
  fit_rfsrc <- rfsrc(reformulate(colnames(X_train), "y_train"), 
                     data = train, 
                     mtry = mtry,
                     ntree = 500, 
                     nodedepth = 10,
                     nodesize = 5,
                     seed = 837363)
  pred_rfsrc <- predict(fit_rfsrc, valid)$predicted
})

system.time({ # 7
  fit_h2o <- h2o.randomForest(
    x = colnames(X_train), 
    y = "y_train",
    train_h2o, 
    ntrees = 500, 
    max_depth = 10, 
    min_rows = 5, 
    mtries = mtry)
  pred_h2o <- as.data.frame(predict(fit_h2o, valid_h2o))$predict
})

# XGB Random Forest
system.time({ # 13
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
system.time({ # 3
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
                       dtrain_lgb,
                       verbose = -1,
                       nrounds = 500)
  
  pred_lgb <- predict(fit_lgb, X_valid)
})

# Evaluate predicitons
pred <- data.frame(
  pred_ranger,
  pred_rfsrc,
  pred_h2o,
  pred_xgb,
  pred_lgb
)

summary(pred) 
cor(pred)
sapply(pred, perf, "mse")
#           pred_ranger pred_rfsrc  pred_h2o  pred_xgb  pred_lgb
# rmse        1.0634036  1.0551927 1.0705415 1.0599327 1.0608430
# r_squared   0.3307503  0.3410454 0.3217357 0.3351119 0.3339694
