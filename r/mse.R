library(ranger)
library(h2o)
library(xgboost)
library(lightgbm)

library(MetricsWeighted)
library(splitTools)

source("r/functions.R")

h2o.init()

# Create data
n <- 1e5
m <- 20
head(full <- make_X(n = n, m = m))
dim(full)
x <- colnames(full)
y <- "y"
full$y <- make_y(full, "mse", strength = 5)
hist(full$y)

# Split into training and validation
idx <- partition(full[[y]], p = c(train = 0.7, valid = 0.3))

train <- full[idx$train, ]
valid <- full[idx$valid, ]

train_h2o <- as.h2o(train)
valid_h2o <- as.h2o(valid)

X_train <- data.matrix(train[, x])
X_valid <- data.matrix(valid[, x])

y_train <- train[[y]]
y_valid <- valid[[y]]

# Random forest settings
mtry <- floor(sqrt(m))
mtry_p <- mtry / m
ntrees <- 100
node_size <- 5
depth <- 20

# Ranger
system.time({ # 15s
  fit_ranger <- ranger(reformulate(x, y),
                       data = train,
                       mtry = mtry,
                       min.node.size = node_size,
                       max.depth = depth, 
                       num.trees = ntrees)
  
  pred_ranger <- predict(fit_ranger, valid)$predictions
})

system.time({ # 39s
  fit_h2o <- h2o.randomForest(
    x = x, 
    y = y,
    train_h2o, 
    ntrees = ntrees, 
    max_depth = depth, 
    min_rows = node_size, 
    mtries = mtry)
  pred_h2o <- as.data.frame(predict(fit_h2o, valid_h2o))$predict
})

# XGB Random Forest
system.time({ # 27
  param_xgb <- list(max_depth = depth,
                    learning_rate = 1,
                    objective = "reg:linear",
                    subsample = 0.63,
                    lambda = 0,
                    alpha = 0,
                    min_child_weight = node_size,
                    # tree_method = "hist",
                    # base_score
                    # max_delta_step
                    colsample_bynode = mtry_p)
  
  dtrain_xgb <- xgb.DMatrix(X_train, label = y_train)
  
  fit_xgb <- xgb.train(param_xgb,
                       dtrain_xgb,
                       nrounds = 1,
                       num_parallel_tree = ntrees)
  
  pred_xgb <- predict(fit_xgb, X_valid)
})


# LGB Random Forest
system.time({ # 10
  param_lgb <- list(boosting = "rf",
                    max_depth = depth,
                    num_leaves = 1000,
                    learning_rate = 1,
                    objective = "regression",
                    bagging_fraction = 0.63,
                    bagging_freq = 1,
                    reg_lambda = 0,
                    reg_alpha = 0,
                    min_data_in_leaf = node_size,
                    colsample_bynode = mtry_p)
  
  dtrain_lgb <- lgb.Dataset(X_train, label = y_train)
  
  fit_lgb <- lgb.train(param_lgb,
                       dtrain_lgb,
                       verbose = -1,
                       nrounds = ntrees)
  
  pred_lgb <- predict(fit_lgb, X_valid)
})

# Evaluate predicitons
pred <- data.frame(
  pred_ranger,
  pred_h2o,
  pred_xgb,
  pred_lgb
)

summary(pred) 
cor(pred)
sapply(pred, perf, "mse")
#           pred_ranger  pred_h2o  pred_xgb  pred_lgb
# rmse        1.0151907 1.0133727 1.0119583 1.0166376
# r_squared   0.2302605 0.2330148 0.2351544 0.2280648
