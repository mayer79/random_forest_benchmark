library(ranger)
library(xgboost)
library(lightgbm)

library(MetricsWeighted)
library(splitTools)

source("r/functions.R")

# Create data
n <- 1e5
m <- 20
head(full <- make_X(n = n, m = m))
dim(full)
x <- colnames(full)
y <- "y"
full$y <- make_y(full, strength = 5, objective = "poisson")
hist(full$y)

# Split into training and validation
idx <- partition(full[[y]], p = c(train = 0.7, valid = 0.3))

train <- full[idx$train, ]
valid <- full[idx$valid, ]

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

# XGB Random Forest
system.time({ # 3
  param_xgb <- list(max_depth = depth,
                    learning_rate = 1,
                    objective = "count:poisson",
                    subsample = 0.63,
                    lambda = 0,
                    alpha = 0,
                    min_child_weight = node_size,
                    # tree_method = "hist",
                    # base_score
                    #max_delta_step = 2,
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
                    objective = "poisson",
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

# Ranger
system.time({ # 14
  fit_ranger <- ranger(reformulate(x, y),
                       data = train,
                       mtry = mtry,
                       min.node.size = node_size,
                       max.depth = depth, 
                       num.trees = ntrees)
  
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
#            pred_xgb   pred_lgb pred_ranger
# deviance   3.303889 1.17156879  1.13913418
# r_squared -1.655372 0.05839744  0.08446549

sessionInfo()

