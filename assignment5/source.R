library(farff)
library(mlr)
library(mlrMBO)
library(caret)

PATH <- "/home/luca/Desktop/Magistrale/Advanced Machine Learning/Assignment 5/fertility.csv"
data <- read.csv(file=PATH, sep=",", header=TRUE)
summary(data)
str(data)

# Pre-process data
data[,1:9] = scale(data[,1:9])
str(data)
data

# Dstribution analysis 
counts <- table(data$Class)
barplot(counts, col = c("green", "blue"), yaxp=c(0, max(counts), 8))

# STEP1 - HPO FOR 2 HYPER-PARAMETERS

initialBudget <- 5
budget <- 20

# Objective function
obj.fun = makeSingleObjectiveFunction(
  name="neuralNet",
  fn = function(x){
    learner <- makeLearner("classif.neuralnet", predict.type = "response", learningrate=x[["learning.rate"]], threshold=x[["threshold"]], hidden=c(4,2))
    task <- makeClassifTask(data=data, target="Class")
    res <- crossval(learner, task, iters=10, stratify = T, keep.pred = T, measures = acc)
    res <- res$aggr
  },
  minimize=F,
  par.set= makeParamSet(
    makeNumericParam("learning.rate", lower=0.01, upper = 0.1),
    makeNumericParam("threshold", lower=0.01, upper=0.1)
  )
)

# Design
design = generateDesign(n = initialBudget, par.set = getParamSet(obj.fun), fun= lhs::maximinLHS)
design$y = apply(design, 1, obj.fun)

# Create surrogate model for Expected Improvement
surr.km = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", control = list(trace = FALSE))
control = makeMBOControl()
control = setMBOControlTermination(control, iters=budget)
control = setMBOControlInfill(control, crit=makeMBOInfillCritEI())
initial_design <- design

run = mbo(obj.fun, design=design, learner=surr.km, control=control, show.info = TRUE)

cat("Best Configuration (EI):", unlist(run$x), "\n")
cat("Best Seen: ", run$y, "\n")

# Create array for intermediate values
bsEI <- numeric(budget+1)
allYs <- getOptPathY(run$opt.path)
allXs <- getOptPathX(run$opt.path)

bsEI[1] <- max(allYs[1:initialBudget])
bsEI[-1] <- allYs[-(1:initialBudget)]
for(i in 2:length(bsEI))
  bsEI[i] <- max(bsEI[i-1], bsEI[i])

# All values
bestY_EI <- max(allYs)
bestY_EI
df <- data.frame(learning.rate = allXs$learning.rate, threshold = allXs$threshold, y = allYs)
df

# Hyperparameters plot
ggplot(data = allXs, mapping = aes(x = learning.rate, y = threshold)) +
  geom_point(data = allXs)

# Create surrogate model for Upper Confidence Bound
design <- initial_design
control = setMBOControlInfill(control, crit = makeMBOInfillCritCB())
run = mbo(obj.fun, design=design, learner=surr.km, control= control, show.info = TRUE)

cat("Best Configuration (UCB):", unlist(run$x), "\n")
cat("Best Seen: ", run$y, "\n")

bsUCB <- numeric(budget+1)
allYs <- getOptPathY(run$opt.path)
allXs <- getOptPathX(run$opt.path)

bsUCB[1] <- max(allYs[1:initialBudget])
bsUCB[-1] <- allYs[-(1:initialBudget)]
for(i in 2:length(bsUCB))
  bsUCB[i] <- max(bsUCB[i-1], bsUCB[i])

# All values
bestY_UCB <- max(allYs)
bestY_UCB
df2 <- data.frame(learning.rate = allXs$learning.rate, threshold = allXs$threshold, y = allYs)
df2

# Hyperparameters plot
ggplot(data = allXs, mapping = aes(x = learning.rate, y = threshold)) +
  geom_point(data = allXs)

# Values for better visualization in the next plot
max1 = max(bsUCB)
max2 = max(bsEI)
max = max(c(max1, max2))

# Plot
plot(0:budget, bsUCB, type="o", lwd=2, col="blue", ylim = range(0.75, 0.9))
lines(0:budget, bsEI, type="o", lwd=2, col="green")
points(0, bsUCB[1], col="red", pch=19)
legend("topleft", legend=c("UCB", "EI"), col=c("blue", "green"), lwd=2)

# Build a Grid Search Design
grid.des = generateGridDesign(par.set = getParamSet(obj.fun), resolution = 5)
grid.des$y = apply(grid.des, 1, obj.fun)
grid.des[which.max(grid.des$y),]

# Best seen
bsGS <- numeric(25)
bsGS[1] <- grid.des[1, ]$y
bsGS[-1] <- grid.des[-1, ]$y
for(i in 2:length(bsGS))
  bsGS[i] <- max(bsGS[i-1], bsGS[i])

# Bidimensional plot for grid search configurations
ggplot(data = grid.des, mapping = aes(x = learning.rate, y = threshold)) +
  geom_point(data = grid.des)

# Build a Random Search Design
random.des = generateRandomDesign(par.set = getParamSet(obj.fun), n = 25)
random.des$y = apply(random.des, 1, obj.fun)
random.des[which.max(random.des$y),]

# Best seen
bsRS <- numeric(25)
bsRS[1] <- random.des[1, ]$y
bsRS[-1] <- random.des[-1, ]$y
for(i in 2:length(bsRS))
  bsRS[i] <- max(bsRS[i-1], bsRS[i])

# Bidimensional plot for random search configurations
ggplot(data = random.des, mapping = aes(x = learning.rate, y = threshold)) +
  geom_point(data = random.des)

# Values for better visualization in the next plot
minRS <- min(bsRS)
minGS <- min(bsGS)
minGRS <- min(c(minRS, minGS))

# Plot
plot(1:25, y=bsGS, type='o', lwd=2, col='orange', ylim=range(minGRS,0.9))
lines(1:25, y=bsRS, type='o', lwd=2, col='red')
legend("topleft", legend=c("Grid Search", "Random Search"), col=c("orange", "red"), lwd=2)

# STEP2 - HPO FOR 4 HYPER-PARAMETERS

obj.fun2=makeSingleObjectiveFunction(
  name="neural_network",
  fn = function(x) {
    learner <- makeLearner("classif.neuralnet", predict.type = "response", learningrate=x[["learning.rate"]], 
                           threshold=x[["threshold"]], hidden=c(x[["layer1"]], x[["layer2"]]))
    task <- makeClassifTask(data = data, target = "Class")
    res <- crossval(learner, task, iters=3, stratify=T, keep.pred = F, measures = acc)
    res <- res$aggr
  },
  minimize=F,
  par.set = makeParamSet(
    makeNumericParam("learning.rate", lower = 0.01, upper = 0.1),
    makeNumericParam("threshold", lower = 0.01, upper = 0.1),
    makeIntegerParam("layer1", lower = 1, upper = 5),
    makeIntegerParam("layer2", lower = 1, upper = 5)
  )
)

# We are going to use just 1 remaining budget to solve the problem described in the documentation
initial <- 10
remaining.budget <- 1

# Random Forest as surrogate model
srr.rf = makeLearner("regr.randomForest", predict.type = "se", ntree=50)
control = makeMBOControl()
control = setMBOControlTermination(control, iters=remaining.budget)
control = setMBOControlInfill(control, crit=makeMBOInfillCritEI())

# Initial design
design = generateDesign(n = initial, par.set = getParamSet(obj.fun2), fun = lhs::improvedLHS)
design$y = apply(design, 1, obj.fun2)
initial_design <- design

# Best Seen placeholder
bsEI <- numeric(101)
bsEI[1] <- max(initial_design$y)

# Manually computing the design using budget 1 and iterating the run 100 times
for(i in 1:100){
  run = mbo(obj.fun2, design=design, learner=srr.rf , control = control, show.info = TRUE)
  
  x = getOptPathX(run$opt.path)
  y = getOptPathY(run$opt.path)
  
  if(max(y)>bsEI[i]){
    bsEI[i+1] <- max(y)
    bestEI <- x 
  }
  else
    bsEI[i+1] <- bsEI[i]
  
  tail_x = tail(x,1)
  tail_y = tail(y,1)
  
  df = data.frame(learning.rate = tail_x$learning.rate, threshold = tail_x$threshold, 
                  layer1 = tail_x$layer1, layer2 = tail_x$layer2, y = tail_y)
  design = df
  print(df)
}

# Best value
bsEI
max(bsEI)

# Upper confidence bound
control = setMBOControlInfill(control, crit=makeMBOInfillCritCB())

# Best Seen placeholder
bsCB <- numeric(101)
bsCB[1] <- max(initial_design$y)
count <- 0

# Manually computing the design using budget 1 and iterating the run 100 times
for(i in 1:100){
  run = mbo(obj.fun2, design=design, learner=srr.rf , control = control, show.info = TRUE)
  
  x = getOptPathX(run$opt.path)
  y = getOptPathY(run$opt.path)
  
  if(max(y)>bsCB[i]){
    bsCB[i+1] <- max(y)
    bestUSB <- x 
  }
  else
    bsCB[i+1] <- bsCB[i]
  
  tail_x = tail(x,1)
  tail_y = tail(y,1)
  
  df = data.frame(learning.rate = tail_x$learning.rate, threshold = tail_x$threshold, 
                  layer1 = tail_x$layer1, layer2 = tail_x$layer2, y = tail_y)
  design = df
  print(df)
  count = count +1
}

bsCB
max(bsCB)
bsUCB <- numeric(count+1)

for(i in 1:101){
  bsUCB[i]=bsCB[i]
}

# Plot
plot(0:100, bsUCB, type="o", lwd=2, col="blue", ylim = range(0.8,0.9))
lines(0:100, bsEI, type="o", lwd=2, col="green")
points(0, bsUCB[1], col="red", pch=19)
legend("bottomright", legend=c("UCB", "EI"), col=c("blue", "green"), lwd=2)

# Print the best values
max(bsCB)
bestUSB
max(bsEI)
bestEI