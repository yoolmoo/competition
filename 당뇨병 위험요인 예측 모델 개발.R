library(caret)
library(ggplot2)
library(lattice)
library(data.table)

# 대용량 데이터 파일 읽기
sweet <- read.csv("/Users/yoolim/Desktop/JoyR/sweet.csv")
sweet$Outcome <- as.factor(sweet$Outcome)

# 읽어들인 데이터 확인
str(sweet)

#트레이닝 테스트 데이터 분할
newdata <- sweet

set.seed(600)
datatotal <- sort(sample(nrow(sweet), nrow(sweet)*0.7))

train <- sweet[datatotal,]
test <- sweet[-datatotal,]

train_x <- train[, 8]
train_y <- train[, 8]

test_x <- test[, 8]
test_y <- test[, 8]

#모형학습
ctrl <- trainControl(method = "repeatedcv", number = 2, repeats = 5 )
customGrid <- expand.grid(k = 1:7)
knnFit <- train( Outcome~.,
                 data = train,
                 method  = "knn",
                 trControl = ctrl,
                 preProcess = c("center", "scale"),
                 tuneGrid = customGrid,
                 metric = "Accuracy")
knnFit
plot(knnFit)

pred_test <- predict(knnFit,newdata = test )
confusionMatrix(pred_test, test$Outcome)

#변수중요도
importance_knn <- varImp(knnFit, scale = FALSE)
plot(importance_knn)
plot(knnFit)

cv <- trainControl(method = "cv", number = 7, verbose = T )

knn.grid = expand.grid(
  .k = 7
)

train.knn <- train(Outcome~., train, method = "knn", trControl = cv, tuneGrid = knn.grid)

train.knn$Outcome

#디시전 트리
library(caret)

sweet <- read.csv(file = "/Users/yoolim/Desktop/JoyR/sweet.csv")
sweet$Outcome <- as.factor(sweet$Outcome)
str(sweet)

set.seed(600)
datatotal <- sort(sample(nrow(sweet), nrow(sweet)*0.7))

train <- sweet[datatotal,]
test <- sweet[-datatotal,]

train_x <- train[, 8]
train_y <- train[, 8]

test_x <- test[, 8]
test_y <- test[, 8]

#트리 만들기

library(tree)

treeRaw <- tree(Outcome~., data = train)
plot(treeRaw)
text(treeRaw)

pred <- predict(treeRaw, train, type = "class")
confusionMatrix(pred, train$Outcome)

#최적 사이즈 찾기
cv_tree <- cv.tree(treeRaw, FUN = prune.misclass)
plot(cv_tree)

#가지치기
prune_tree <- prune.misclass(treeRaw, best = 10)
plot(prune_tree)
text(prune_tree, pretty = 0)

#Accuracy 확인
pred <- predict(prune_tree, test, type = "class")
confusionMatrix(pred, test$Outcome)

#랜덤 포레스트

library(caret)

ctrl <- trainControl(method = "repeatedcv", repeats = 5)
rfFit <- train(Outcome~.,
               data = train,
               method = "rf",
               trControl = ctrl,
               preProcess = c("center","scale"),
               metric = "Accuracy")
rfFit
plot(rfFit)

pred_test <- predict(rfFit, newdata = test)
confusionMatrix(pred_test, test$Outcome)

importance_rf <- varImp(rfFit, scale = FALSE)
plot(importance_rf)

table(pred_test, test$Outcome)
