setwd("~/Downloads/ml-latest-small")

rm(list = ls())
gc()

library(data.table)
library(dplyr)
library(tidyr)
library(reshape2)
library(Metrics)

schema <- c("userId" = "character",
            "movieId" = "character",
            "rating" = "numeric",
            "timestamp" = "character")
df <- fread("ratings.csv", colClasses = schema)

## label train (+1) / val (-1)
df$label <- 1
df$label[sample(1:nrow(df),0.3*nrow(df))] <- (-1)

# Y = U X I
Y = dcast(df, userId ~ movieId, value.var = 'rating', fill = 0)
Y <- as.matrix(Y[2:ncol(Y)])

R <- dcast(df,userId ~ movieId, value.var = 'label', fill = 0)
R <- as.matrix(R[2:ncol(R)])
R.train <- R
R.train[R.train<1] <- 0
R.val <- (-R)
R.val[R.val<1] <- 0

loss <- function(Y,U,I,R,lambda){
  return((sum(R*(Y-U %*% t(I))**2) + lambda*sum(U**2) + lambda*sum(I**2))/sum(R)/2)
}

grad_u <- function(Y,U,I,R,lambda) (-(R*(Y - U%*%t(I))) %*% I + lambda*abs(U))/sum(R)
grad_i <- function(Y,U,I,R,lambda) (-t(R*(Y - U%*%t(I))) %*% U + lambda*abs(I))/sum(R)

gd <- function(Y,U,I,R, lambda, alpha, maxIter,thresh = 1e-4){
  loss0 <- loss(Y,U,I,R[[1]],lambda)
  alpha0 <- alpha
  gr_u <- grad_u(Y,U,I,R[[1]],lambda)
  gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  for(i in 1:maxIter){
    U <- U - gr_u * alpha
    I <- I - gr_i * alpha
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    if (abs(loss1-loss0)<thresh | loss1 > loss0) break
    cat('iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')
    loss0 <- loss1
    gr_u <- grad_u(Y,U,I,R[[1]],lambda)
    gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  }
  return(list(U = U,I = I,loss = loss1,iter = i, rmse.train = rmse.train, rmse.val = rmse.val))
}

als <- function(Y,U,I,R,lambda, alpha, maxIter,thresh = 1e-4){
  loss0 <- loss(Y,U,I,R[[1]],lambda)
  alpha0 <- alpha
  gr_u <- grad_u(Y,U,I,R[[1]],lambda)
  gr_i <- grad_i(Y,U,I,R[[1]],lambda)
  for(i in 1:maxIter){
    
    # update User Matrix
    U <- U - gr_u * alpha
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    if (abs(loss1-loss0)<thresh | loss1 > loss0) break
    cat('U iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')    
    gr_u <- grad_u(Y,U,I,R[[1]],lambda)
    loss0 <- loss1
    
    # update Item Matrix
    I <- I - gr_i * alpha
    loss1 <- loss(Y,U,I,R[[1]],lambda)
    pred <- (U%*%t(I))
    rmse.train <- rmse(pred[R[[1]]>0],Y[R[[1]]>0])
    rmse.val <- rmse(pred[R[[2]]>0],Y[R[[2]]>0])
    if (abs(loss1-loss0)<thresh | loss1 > loss0) break
    cat('I iter',i,': loss',loss1,'alpha',alpha,'rmse.train',rmse.train,
        'rmse.val',rmse.val,'\n')    
    gr_i <- grad_i(Y,U,I,R[[1]],lambda)
    loss0 <- loss1
    
  }
  return(list(U = U,I = I,loss = loss1,iter = i, rmse.train = rmse.train, rmse.val = rmse.val))
}

m <- nrow(Y)
n <- ncol(Y)
k <- 5
## initiaize U and I
U = matrix(runif(m*k),m,k)
I = matrix(runif(n*k),n,k)

gd.time <- system.time(res.gd <- gd(Y,U,I,list(R.train,R.val),0.3,30,100))
als.time <- system.time(res.als <- als(Y,U,I,list(R.train,R.val),0.3,30,50))

save(gd.time,als.time,res.gd, file = 'model_result')
