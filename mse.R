library(xgboost)
library(lightgbm)
library(ranger)

set.seed(1)
n <- 1000
set.seed(1)
n <- 1000
x1 <- seq_len(n)
x2 <- rnorm(n)
x3 <- rexp(n)
x4 <- runif(n)
X <- cbind(x1, x2, x3, x4)
y <- rnorm(n, x1 / 1000 + x2 / 10 + x3 / 5)

# XGB Random Forest
param_xgb <- list(max_depth = 10,
                  learning_rate = 1,
                  objective = "reg:linear",
                  subsample = 0.63,
                  lambda = 0,
                  alpha = 0,
                  colsample_bynode = 1/3)

dtrain_xgb <- xgb.DMatrix(X, label = y)

fit_xgb <- xgb.train(param_xgb,
                     dtrain_xgb,
                     nrounds = 1,
                     num_parallel_tree = 500)


# LGB Random Forest
param_lgb <- list(boosting = "rf",
                  max_depth = 10,
                  num_leaves = 1000,
                  learning_rate = 1,
                  objective = "regression",
                  bagging_fraction = 0.63,
                  bagging_freq = 1,
                  reg_lambda = 0,
                  reg_alpha = 0,
                  min_data_in_leaf = 1,
                  colsample_bynode = 1/3)

dtrain_lgb <- lgb.Dataset(X, label = y)

fit_lgb <- lgb.train(param_lgb,
                     dtrain_lgb,
                     nrounds = 500)

# True Random Forest
fit_rf <- ranger(y = y, 
                 x = X, 
                 max.depth = 10, 
                 num.trees = 500)

# Evaluate predicitons
pred <- data.frame(
  pred_xgb = predict(fit_xgb, X),
  pred_lgb = predict(fit_lgb, X),
  pred_rf = predict(fit_rf, X)$predictions
)

summary(pred) 
cor(pred)

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

rmse(y, pred$pred_xgb) # 0.6170593
rmse(y, pred$pred_lgb) # 0.6047725
rmse(y, pred$pred_rf)  # 0.6003823

sessionInfo()
