#-- function for binary data generation. A and B are the covariance matrices and m0 and m1 are vector means
gen_bin_data = function(n, m0, m1, A, B) {
  library(MASS)
  class_0 = mvrnorm(n/2, m0, sqrt(A))
  class_1 = mvrnorm(n/2, m1, sqrt(B))
  df_class_0 = cbind(class_0, Y = 0)
  df_class_1 = cbind(class_1, Y = 1)
  df = as.data.frame(rbind(df_class_0, df_class_1))
  df$Y = as.factor(df$Y)
  names(df)[1] = "X1"
  names(df)[2] = "X2"
  
  #-- Randomly shuffle
  rows = sample(nrow(df))
  df = df[rows,]
  row.names(df) = NULL
  return(df)
}

#-- Function to calculate misclassification error on a classifier
calc_class_err = function(actual, predicted) {
  mean(actual != predicted)
}

#-- AdaBoosting function
boost_my_tree = function(formula, data, max.depth = 1, B = 199){
  library(rpart)
  #-- Recode the class labels to get the outcome in shape {-1,1}
  n = nrow(data)
  cl.var = all.vars(formula)[[1]]
  labels = unique(data[[cl.var]])

  labels.new = rep(NA, n)
  labels.new[data[cl.var] == labels[1]] = 1
  labels.new[data[cl.var] == labels[2]] = -1
  data.train = data
  data.train[cl.var] = as.factor(labels.new)
  
  #-- Initialization
  weights <- rep(1/n, n)
  learners <- list("")
  agg.weights <- rep(0, B)
  data.train$weights <- rep(0, n)
  
  #-- The boosting loop
  for (i in 1:B){
    
    #-- Train the learner
    data.train$weights <- weights
    learners[[i]] <- rpart(formula, data.train, weights, method = "class", 
                           control = rpart.control(maxdepth = max.depth, 
                                                   minsplit = 2, 
                                                   minbucket = 1, cp = 0))
    #-- Calculate the error
    forcasted <- predict(learners[[i]], data.train, "class")
    err <- mean(weights * as.numeric(forcasted != data.train[[cl.var]]))

    #-- Calculate aggregating weights and update weights
    agg.weights[i] <- log((1 - err) / err)
    weights <- weights * exp(agg.weights[i] * ifelse(forcasted != data.train[[cl.var]], 1, 0))
    weights <- weights / sum(weights)
  }
  # Construct the resulting structure
  adatree <- structure(
    list(learners = learners, 
         weights = agg.weights, 
         labels = labels), 
    .Names = c("learners", "weights", "labels")
  )
  return(adatree)
}

#-- Function to make predictions using the constructed Adaboosted tree
adaboost.tree.classify <- function(adaboost, objects){
  # Get the scores
  sums <- rep(0, nrow(objects))
  for (i in 1:length(adaboost$learners)){
    curForcast <- predict(adaboost$learners[[i]], objects, "class")
    sums <- sums + 
      as.numeric(levels(curForcast))[as.numeric(curForcast)] * 
      adaboost$weights[i]
  }
  # Return the classes
  return (adaboost$labels[(sums < 0) + 1])
}

#-- Function for distributional setting number 10 (normal exponential)
Mix2 <- function(numLearn, numTest){
  library("MASS")
  l1 <- mvrnorm(numLearn, c(0,0), matrix(c(1,0,0,1), nrow = 2, ncol = 2, byrow = TRUE))
  l2 <- cbind(rexp(numLearn, 1), rexp(numLearn, 1))
  t1 <- mvrnorm(numTest, c(0,0), matrix(c(1,0,0,1), nrow = 2, ncol = 2, byrow = TRUE))
  t2 <- cbind(rexp(numTest, 1), rexp(numTest, 1))
  learnData <- rbind(cbind(l1, rep(0, numLearn)), cbind(l2, rep(1, numLearn)))
  testData <- rbind(cbind(t1, rep(0, numTest)), cbind(t2, rep(1, numTest)))
  rez <- list(learn = learnData, test = testData)
  return(rez)
}

#-- Function for distributional setting number 8 (exponential location scale)
Exponential2 <- function(numLearn, numTest){
  library("MASS")
  l1 <- cbind(rexp(numLearn, 1), rexp(numLearn, 0.5))
  l2 <- cbind(rexp(numLearn, 0.5) + 1, rexp(numLearn, 1) + 1)
  t1 <- cbind(rexp(numTest, 1), rexp(numTest, 0.5))
  t2 <- cbind(rexp(numTest, 0.5) + 1, rexp(numTest, 1) + 1)
  learnData <- rbind(cbind(l1, rep(0, numLearn)),cbind(l2, rep(1, numLearn)))
  testData <- rbind(cbind(t1, rep(0, numTest)),cbind(t2, rep(1, numTest)))
  colnames(learnData) = colnames(testData) = c("X1","X2", "Y")
  rez <- list(learn = learnData, test = testData)
  return(rez)
}