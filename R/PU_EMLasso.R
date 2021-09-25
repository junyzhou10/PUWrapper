#' @title PU learning solved by regularized (LASSO) EM algorithm
#' @description General parametric framework for the PU learning with variable selection embedded
#' @param Z The observed binary outcomes under the PU setting, i.e. the 1s represent the positives instances with certainty and the 0s represent the unlabeled instances
#' @param X The nxp design matrix, where n is the number of instances and the p is the number of covariates (no bias/intercept term, i.e., the whole 1 column)
#' @param PI The population prevalence, i.e., Pr(y==1), where y is the true binary outcomes
#' @param lambda.sel c('lambda.min', 'lambda.1se'), used for selecting lambda in cv.glmnet. The default is 'lambda.min'
#' @param intercept If intercept should be considered. The default is TRUE
#' @param eps Convergence threshold for EM algorithm
#' @param max.iter Maximum number of iterations for the EM algorithm
#' @param ... Other arguments for glmnet/cv.glmnet
#' @details This is the PU learning method proposed in the paper: Ward, G., Hastie, T., Barry, S., Elith, J., & Leathwick, J. R. (2009). Presence‐only data and the EM algorithm. Biometrics, 65(2), 554-563
#' @return A list object containing estimated probability, estimated/selected parameters, and the corresponding penalty coefficient lambda (based on glmnet).
#' @examples
#' set.seed(123)
#' res = PU_EMLasso(sample_dat$Z, sample_dat$X, sample_dat$PI, alpha = 1)
#' # plot(res$Est.Pr, sample_dat$true.prob)
#' @import glmnet utils
#' @importFrom stats coef
#' @export


PU_EMLasso <- function(Z, X, PI, lambda.sel = 'lambda.min', intercept = TRUE, eps = 1e-3, max.iter = 1000, ...) {
  n.var = ncol(X)
  n_l   = sum(Z==1) # positive instances
  n_u   = sum(Z==0) # unlabeled instances
  b     = log((n_l + PI*n_u)/(PI*n_u)) # a constant adjustment to the mean to account for the PU framework

  # warm start by fitting logistic regression model directly using Z as outcome
  fit.y <- cv.glmnet(x = X, y = Z, family = "binomial", ...)
  betas <- coef(fit.y, s = lambda.sel)
  if (intercept) {
    Xbeta = cbind(1, X) %*% betas
  } else {
    Xbeta = X %*% betas
  }
  Loglik <- mean(Xbeta*Z - log(1+exp(Xbeta)))
  Obj    <- -Loglik + fit.y[[lambda.sel]]*sum(abs(betas))
  beta0  <- betas

  # start iterations until converge
  iter = 1
  progress.ind <- rep(NA, times = max.iter)
  progress.bar <- txtProgressBar(0, length(progress.ind), style = 2)
  while (TRUE) {
    setTxtProgressBar(progress.bar, iter)
    # print(c(Loglik, Obj, betas@x))
    # print("=")
    y.hat = (1/(1+exp(-Xbeta)))^(1-Z)

    # data augmentation to deal with non-binary outcomes: y.hat, so that glmnet could work
    Y1 = rep(1, nrow(X)); w1 = as.numeric(y.hat)
    Y2 = rep(0, nrow(X)); w2 = 1-as.numeric(y.hat)
    # use offset to adjust the intercept with amount b
    fit.y <- cv.glmnet(x = rbind(X,X), y = c(Y1, Y2), family = "binomial", weights = c(w1,w2), offset = rep(b, 2*nrow(X)), ...)
    betas <- coef(fit.y, s = lambda.sel)
    if (intercept) {
      Xbeta = cbind(1, X) %*% betas
    } else {
      Xbeta = X %*% betas
    }
    Loglik <- mean((Xbeta+b)*Z - log(1+exp(Xbeta+b)))
    Obj    <- -Loglik + fit.y[[lambda.sel]]*sum(abs(betas))
    if (sum(abs(beta0 - betas))<eps*diff(betas@p) | iter >= max.iter){
      break
    } else {
      beta0 <- betas
      iter = iter + 1
    }
  }
  cat("Done!")
  close(progress.bar)

  return(list(Est.Pr = (1/(1+exp(-Xbeta))), betas = betas, lambda = fit.y[[lambda.sel]]))
}
