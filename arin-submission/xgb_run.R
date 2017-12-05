library(xgboost)
library(data.table)
library(purrr)

# note we are using the new file given by Garett here, change accordingly
train_data  <- fread('../input/preprocessed_train.csv', stringsAsFactors = T)
train_data$type <- 1

test_data  <- fread('../input/preprocessed_test.csv', stringsAsFactors = T)
test_data$type <- 0
ids = test_data$id
test_data$id = NULL #only for Garett's preprocessed file

train_df_model <- train_data
y_train <- train_df_model$target

train_df_model$target <- NULL

# Row binding feature engineering
train_test <- rbindlist(l = list(train_df_model, test_data),use.names = T)
#ntrain <- nrow(train_df_model)
features <- names(train_data)


#convert character into integer
for (f in features) {
  if (is.character(train_test[[f]])) {
    levels = sort(unique(train_test[[f]]))
    train_test[[f]] = as.integer(factor(train_test[[f]],levels = levels))
  }
}

#splitting whole data back again
train_x <- train_test[type==1,]
test_x <- train_test[type==0,]
train_x$type <- NULL
test_x$type <- NULL

#convert into numeric for XGBoost implementation
train_x[] <- map(train_x, as.numeric)
test_x[] <- map(test_x, as.numeric)
dtrain <- xgb.DMatrix(as.matrix(train_x),label = y_train)
dtest <- xgb.DMatrix(as.matrix(test_x))

# Model Run ------------------

set.seed(1234)
rm(test_x,train_df_model,train_test)

xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- MLmetrics::NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

##xgboost parameters
xgb_params <- list(booster = "gbtree" 
                   , objectve = "binary:logistic"
                   , eta=0.02                       
                   , gamma=1
                   , max_depth=7
                   , subsample=0.7
                   , colsample_bytree = 0.8
                   , min_child_weight = 1
                   , base_score=median(y_train)
)

#tuning - tobe run locally
# xgbcv <- xgb.cv(params = xgb_params
#                , data = dtrain
#                , nrounds = 2000
#                , nfold = 4
#               , print_every_n = 20
#               , early_stopping_rounds = 20
#               , maximize = F
#                , prediction = F
#                , showsd = T
# )

#train model
start.time <- Sys.time() #clock running time
gb_dt <- xgb.train(params = xgb_params
                   , data = dtrain
                   , nrounds = 630
                   , feval = xgb_normalizedgini
                   , maximize = TRUE
                   , print_every_n = 20
                   , early_stopping_rounds = 10
                   , verbose = 1
                   , watchlist = list(train=dtrain)
)
end.time <- Sys.time()
xgb.time.taken <- round(end.time - start.time,2)

# fit the model and save output ---------
test_preds <- predict(gb_dt, dtest, type = "prob")

# sub <- data.frame(id = test_data$id, target = test_preds)
sub <- data.frame(id = ids, target = test_preds) #when using preprocessed dataset from Gprep
sub$target <- abs(sub$target)
write.csv(sub, '../output/xgbt_submission.csv', row.names = FALSE)