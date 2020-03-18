library(ranger)
library(xgboost)
library(lightgbm)

library(MetricsWeighted)
library(splitTools)

source("r/functions.R")

# Create data
head(X <- make_X())
head(y <- make_y(X, "poisson", strength = 1))
barplot(table(y))

mtry <- floor(sqrt(ncol(X)))
mtry_p <- mtry / ncol(X)

# Split into training and test
idx <- partition(y, p = c(train = 0.7, valid = 0.3))
X_train <- X[idx$train, ]
y_train <- y[idx$train]
X_valid <- X[idx$valid, ]
y_valid <- y[idx$valid]

# XGB Random Forest
system.time({ # 12
  param_xgb <- list(max_depth = 10,
                    learning_rate = 1,
                    objective = "count:poisson",
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
                    objective = "poisson",
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


# Evaluate predicitons
pred <- data.frame(
  pred_xgb,
  pred_lgb,
  pred_ranger
)

summary(pred) 
cor(pred)
sapply(pred, perf, "poisson")
#             pred_xgb   pred_lgb pred_ranger
# deviance  1.05753011 1.04356883  1.02037717
# r_squared 0.04407729 0.05669717  0.07766058

sessionInfo()