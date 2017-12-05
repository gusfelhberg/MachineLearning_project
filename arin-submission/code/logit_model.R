library(caret)
library(randomForest)
library(dplyr)

train <- data.table::fread('../input/train.csv')
# head(train)
# tail(train)

corrMatrix <- cor(train[,3:ncol(train)])

highCorr <- findCorrelation(corrMatrix, cutoff=0.75, verbose = FALSE, names = TRUE)

# print(highCorr)

# We'll also eliminate predictors with very little variance
nzv <- nearZeroVar(train, freqCut = 95/5, saveMetrics = FALSE,
  names = TRUE, foreach = FALSE, allowParallel = TRUE)

# print(nzv)

#drop cols
dropList <- list(
    "ps_ind_05_cat",
    "ps_ind_10_bin",
    "ps_ind_11_bin",
    "ps_ind_12_bin",
    "ps_ind_13_bin",
    "ps_ind_14",
    "ps_reg_03",
    "ps_car_10_cat",
    "id"
)

for (d in dropList){
    train[, eval(d) := NULL]
}

# Based on the column names, some of the predictors are categorical. Let's one-hot encode them.
catList <- grep("_cat", colnames(train), value = TRUE)

train[, catList] <- lapply(train[, catList, with=FALSE], factor)

dmy <- dummyVars(
    "~ps_ind_02_cat + 
    ps_ind_04_cat + 
    ps_car_01_cat + 
    ps_car_02_cat + 
    ps_car_03_cat + 
    ps_car_04_cat + 
    ps_car_06_cat + 
    ps_car_07_cat + 
    ps_car_08_cat + 
    ps_car_09_cat + 
    ps_car_11_cat",
    data = train,
    fullRank = TRUE
    )

encoded <- data.frame(predict(dmy, newdata = train)) 

# Bind new encoded columns to train
train <- cbind(train,encoded)

# Save RAM
rm(encoded)

# Drop original categorical features
#train[, which(catList == TRUE) := NULL] # This throws an error, come back to it
train[, ps_ind_02_cat := NULL]
train[, ps_ind_04_cat := NULL]
train[, ps_car_01_cat := NULL]
train[, ps_car_02_cat := NULL]
train[, ps_car_03_cat := NULL]
train[, ps_car_04_cat := NULL]
train[, ps_car_06_cat := NULL]
train[, ps_car_07_cat := NULL]
train[, ps_car_08_cat := NULL]
train[, ps_car_09_cat := NULL]
train[, ps_car_11_cat := NULL]

# Change -1 to NA
train[train == -1] <- NA
paste0("% NA after transforming -1... ", sum(is.na(train))/prod(dim(train)))

# Drop all columns with >= x% NA
vectordrop <- train[, lapply( train, function(m) sum(is.na(m)) / length(m) ) >= .05 ]

# Only x columns have >= 5% NA ... let's drop them
train[, names(which(vectordrop == TRUE)) := NULL]

# Impute missing values using medians
imp_values <- preProcess(train,
                        method = c("medianImpute")
                        )

train_imputed <- predict(imp_values, train)

#Now we split above sample data into train and test data ##
set.seed(123)
# making a train index
train_index <- sample(c(TRUE, FALSE), replace = TRUE, size = nrow(train_imputed), prob = c(0.7, 0.3))

# split the data according to the train index
training <- as.data.frame(train_imputed[train_index, ])
testing <- as.data.frame(train_imputed[!train_index, ])

# ----- Logistic Regression ---------------------------------

# estimate logistic regression model on training data

start.time <- Sys.time() #clock running time
logmod <- glm(as.factor(target) ~ ., data = training, family = binomial(link = 'logit'))
end.time <- Sys.time()
time.taken2 <- round(end.time - start.time,2)

preds <- predict(logmod, newdata = testing, type = "response")

#---- some inference on the logit model here -------------------

# Note that The confusion matrix is calculated at a specific point determined 
# by the cutoff on the votes. Depending on your needs, i.e., better precision 
# (reduce false positives) or better sensitivity (reduce false negatives) you 
# may prefer a different cutoff.

roc_data <- data.frame(
    p0.3 = ifelse(preds > 0.3, 1, 0),
    p0.2 = ifelse(preds > 0.2, 1, 0),
    p0.1 = ifelse(preds > 0.1, 1, 0),
    p0.05 = ifelse(preds > 0.05, 1, 0),
    p0.04 = ifelse(preds > 0.04, 1, 0),
    p0.03 = ifelse(preds > 0.03, 1, 0),
    p0.02 = ifelse(preds > 0.02, 1, 0),
    p0.01 = ifelse(preds > 0.01, 1, 0)
    )
# true positive (hit) rate
tpr <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 1 & res$pred == 1) / sum(actual == 1)
}

# false positive rate
fpr <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 0 & res$pred == 1) / sum(actual == 0)
}

# true positive
tp <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 1 & res$pred == 1)
}

# true negative
tn <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 0 & res$pred == 0)
}

# false positive
fp <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 0 & res$pred == 1)
}

# false negative
fn <- function(pred, actual) {
    res <- data.frame(pred, actual)
    sum(res$actual == 1 & res$pred == 0)
}


# get actual values from testing data
actual <- testing$target

# Lets now see the effect on accuracy on different threshold values.
library(dplyr)
library(tidyr)
roc_data <- roc_data %>% 
    gather(key = 'threshold', value = 'pred') %>% 
    group_by(threshold) %>%
    summarize(tp = tp(pred, actual = actual), 
              fp = fp(pred, actual = actual),
              fn = fn(pred, actual = actual),
              tn = tn(pred, actual = actual)
              )

roc_data$accuracy = (roc_data$tp + roc_data$tn) / (roc_data$tp + roc_data$tn + roc_data$fp + roc_data$fn)
roc_data$recall = (roc_data$tp) / (roc_data$tp + roc_data$fn)
roc_data$prediction = (roc_data$tp) / (roc_data$tp + roc_data$fp)


# plot
library(reshape2)
plot_data <- melt(roc_data[c(1,6:8)], id.vars="threshold", value.name="val", variable.name="type")
ggplot(data=plot_data, aes(x=threshold, y=val, group = type, colour = type)) +
    geom_line() +
    geom_point( size=2, shape=21, fill="white")
    
