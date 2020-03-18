make_X <- function(n = 1000, m_int = 5, m_float = 5, seed = 1) {
  set.seed(seed)
  X_int <- matrix(sample(0:100, n * m_int, replace = TRUE), ncol = m_int,
                  dimnames = list(NULL, paste0("X", seq_len(m_int))))
  X_float <- matrix(runif(n * m_float), ncol = m_float, 
                    dimnames = list(NULL, paste0("X", seq_len(m_float))))
  cbind(X_int, X_float)
}

make_y <- function(X, objective = c("mse", "poisson"), strength = 5) {
  objective <- match.arg(objective)
  beta <- seq_len(ncol(X)) - 1
  mu <- X %*% beta
  mu_scaled <- strength * mu / max(mu)
  switch(objective, 
         mse = rnorm(nrow(X), mu_scaled),
         poisson = rpois(nrow(X), mu_scaled))
}

perf <- function(pred, objective = c("mse", "poisson")) {
  objective <- match.arg(objective)
  
  if (objective == "mse") {
    return(c(rmse = rmse(y_valid, pred),
             r_squared = r_squared(y_valid, pred)))
  }
  if (objective == "poisson") {
    return(c(deviance = deviance_poisson(y_valid, pred),
             r_squared = r_squared_poisson(y_valid, pred)))
  }
}
