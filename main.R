library(nnet)
library(class)
library(kernlab)
library(kknn)
library(ggplot2)
library(doParallel)
library(bigstatsr)
library(xtable)

source("utils.R")

set.seed(12323455)

link <- "https://raw.githubusercontent.com/andersonara/datasets/master/wall-robot-navigation.csv"
data <- read.csv(url(link), sep = ";")
data$Y <- factor(data$Y, ordered = FALSE)
#Separação - Treinamento e teste
size_tr <- round(.80 * nrow(data))
train_indx <- sample(1:nrow(data), size_tr, replace = FALSE)
train <- data[train_indx, ]
test <- data[-train_indx, ]

#Realização de validação holdout repetido
n_holdout <- 500
hold_prop <- .2
test_holdout_size <- round(hold_prop*nrow(train))

#K_grid
k_grid <- 1:20
mcc_vec <- numeric(length(k_grid))
for(i in 1:length(k_grid)){
  mcc_avg <- numeric(n_holdout)
  mcc_avg <- mcsapply(X = 1:n_holdout, mc.cores = detectCores(), function(x){
    test_indx_holdout <- sample(1:nrow(train), test_holdout_size,
                                replace = FALSE)
    holdout_train <- train[-test_indx_holdout, ]
    holdout_test <- train[test_indx_holdout, -3]
    holdout_test_labels <- train[test_indx_holdout, 3]
    model <- kknn(Y~.,train = holdout_train, test = holdout_test,
                  kernel = "rectangular", k = k_grid[i])
    Metrics::accuracy(labs, fitted(model))
  }, simplify = T)
  mcc_vec[i] <- mean(mcc_avg)
}
#Validação LOOCV
mcc_vec_loocv <- numeric(length(k_grid))
for(i in 1:length(k_grid)){
  #n_cores <- detectCores()
  #cluster <- makeCluster(n_cores)
  #registerDoParallel(cluster)
  #mcc_avg <- FBM(nrow = nrow(train), ncol = 1)
  mcc_avg <- numeric(nrow(train))
  obs <- factor(rep(1,nrow(train)), levels = 1:4)
  prd <- factor(rep(1,nrow(train)), levels = 1:4)
  sapply(X = 1:nrow(train), function(x){
    data_tr <- train[-x,] 
    data_test <- train[x,]
    model <- kknn(Y~.,train = data_tr, test = data_test, kernel = "rectangular", 
                  k = k_grid[i])
    obs[x] <<- data_test$Y
    prd[x] <<- fitted(model)
  })
  #stopCluster(cluster)
  print(i)
  mcc_vec_loocv[i] <- Metrics::accuracy(labs, fitted(model))
}
#Validação holdout simples
mcc_simple_holdout <- numeric(length(k_grid))
ind_holdout <- sample(1:nrow(train), round(.8*nrow(train)), replace = FALSE)
data_tr <- train[-ind_holdout,] 
data_test <- train[ind_holdout, -3]
labs <- train[ind_holdout, 3]
for(i in 1:length(k_grid)){
  model <- kknn(Y~., train = data_tr, test = data_test, kernel = "rectangular", 
                k = k_grid[i])
  mcc_simple_holdout[i] <- Metrics::accuracy(labs, fitted(model))
}
#K selecionado
selected_k_holdout <- k_grid[which.max(mcc_vec)]
selected_k_simpleholdout <- k_grid[which.max(mcc_simple_holdout)]
selected_k_loocv <- k_grid[which.max(mcc_vec_loocv)]

# Rodando para holdout
knn_model_fit <- kknn(Y~.,train = train, test = train, k = selected_k_holdout,
                      kernel = "rectangular")
knn_model_final <- kknn(Y~.,train = train, test = test, k = selected_k_holdout,
                        kernel = "rectangular")
mcc_custom(test$Y, fitted(knn_model_final))
Metrics::accuracy(test$Y, fitted(knn_model_final))
#Rodando para loocv
knn_model_fit_loocv <- kknn(Y~.,train = train, test = train, 
                            k = selected_k_loocv,
                            kernel = "rectangular")
knn_model_final_loocv <- kknn(Y~.,train = train, test = test, 
                              k = selected_k_loocv,
                              kernel = "rectangular")
mcc_custom(test$Y, fitted(knn_model_final_loocv))
Metrics::accuracy(test$Y, fitted(knn_model_final_loocv))

#Rodando para holdoutsimples
knn_model_fit_simple <- kknn(Y~.,train = train, test = train, 
                            k = selected_k_simpleholdout,
                            kernel = "rectangular")
knn_model_final_simple <- kknn(Y~.,train = train, test = test, 
                              k = selected_k_simpleholdout,
                              kernel = "rectangular")
mcc_custom(test$Y, fitted(knn_model_final_simple))
Metrics::accuracy(test$Y, fitted(knn_model_final_simple))

#Holdout simples
#Fronteira de decisão - dados de treinamento
x1_grid <- seq(min(train$X1), max(train$X1), length.out = 150)
x2_grid <- seq(min(train$X2), max(train$X2), length.out = 150)
dtgrid <- expand.grid(X1 = x1_grid, X2 = x2_grid) %>%
  rbind(train[, c("X1", "X2")])
mdgrid <- kknn(Y~., train = train, test = dtgrid, k = selected_k_simpleholdout,
               kernel = "rectangular")
knngrid <- cbind(dtgrid, fitted(mdgrid))
colnames(knngrid)[3] <- "Ygrid"
#Versão GGPlot
ggplotdataknngrid <- merge(knngrid, train, by = c("X1", "X2"), all.x = T)
ggplot(rename(ggplotdataknngrid, `Classe Real` = Ygrid)) +
  geom_point(aes(x = X1, y = X2, color = `Classe Real`), size = .2) + 
  theme_bw() +
  geom_point(data = filter(ggplotdataknngrid, !is.na(Y)),
             aes(x = X1, y = X2, color = Y)) + 
  scale_color_brewer(palette = "Dark2") + 
  labs(title = "Fronteiras de decisão: Regressão multinomial via KNN",
       subtitle = "K selecionado via LOOCV e Holdout Simples: 1")

#Holdout repetido - dados de treinamento
x1_grid <- seq(min(train$X1), max(train$X1), length.out = 150)
x2_grid <- seq(min(train$X2), max(train$X2), length.out = 150)
dtgrid <- expand.grid(X1 = x1_grid, X2 = x2_grid) %>%
  rbind(train[, c("X1", "X2")])
mdgrid2 <- kknn(Y~., train = train, test = dtgrid, k = selected_k_holdout,
               kernel = "rectangular")
knngrid <- cbind(dtgrid, fitted(mdgrid2))
colnames(knngrid)[3] <- "Ygrid"
#Versão GGPlot
ggplotdataknngrid <- merge(knngrid, train, by = c("X1", "X2"), all.x = T)
ggplot(rename(ggplotdataknngrid, `Classe Real` = Ygrid)) +
  geom_point(aes(x = X1, y = X2, color = `Classe Real`), size = .2) + 
  theme_bw() +
  geom_point(data = filter(ggplotdataknngrid, !is.na(Y)),
             aes(x = X1, y = X2, color = Y)) + 
  scale_color_brewer(palette = "Dark2") + 
  labs(title = "Fronteiras de decisão: Regressão multinomial via KNN",
       subtitle = "K selecionado via Holdout repetido: 5")

#### AJUSTE: Regressão multinomial pelo pacote nnet
mdmult <- multinom(Y~., data = train)
summary(mdmult)
multinomprd <- predict(mdmult, test)
mcc_custom(test$Y, multinomprd)
Metrics::accuracy(test$Y, multinomprd)
## Fronteiras de decisão
mdmultgrid <- cbind(dtgrid, predict(mdmult, dtgrid))
colnames(mdmultgrid)[3] <- "Classe Real"
## Versão GGPlot
ggplotdatamultgrid <- merge(mdmultgrid, train, by = c("X1", "X2"), all.x = T)
ggplot(ggplotdatamultgrid) +
  geom_point(aes(x = X1, y = X2, color = `Classe Real`), size = .2) + 
  theme_bw() +
  geom_point(data = filter(ggplotdatamultgrid, !is.na(Y)),
             aes(x = X1, y = X2, color = Y)) + 
  scale_color_brewer(palette = "Dark2") + 
  labs(title = "Fronteiras de decisão: Regressão logística multinomial via nnet")
# Métricas
# acc, F-1 Score, mcc, tudo. 

#Holdout simples
acc_simple <- Metrics::accuracy(test$Y, fitted(knn_model_final_simple))
f1_simple <- Metrics::f1(test$Y, fitted(knn_model_final_simple))
mcc_simple <- mcc_custom(test$Y, fitted(knn_model_final_simple))

#Holdout repetido
acc_repeat <- Metrics::accuracy(test$Y, fitted(knn_model_final))
f1_repeat <- Metrics::f1(test$Y, fitted(knn_model_final))
mcc_repeat <- mcc_custom(test$Y, fitted(knn_model_final))

#LOOCV
acc_loocv <- Metrics::accuracy(test$Y, fitted(knn_model_final_loocv))
f1_loocv <- Metrics::f1(test$Y, fitted(knn_model_final_loocv))
mcc_loocv <- mcc_custom(test$Y, fitted(knn_model_final_loocv))

#REGRESSÃO LOGISTICA MULTINOMIAL
acc_multinom <- Metrics::accuracy(test$Y, multinomprd)
f1_multinom <- Metrics::f1(test$Y, multinomprd)
mcc_multinom <- mcc_custom(test$Y, multinomprd)

tabela <- data.frame(Modelo = c("KNN - Holdout Simples", "KNN - Holdout Repetido", 
                      "KNN - LOOCV", "Reg Logística Multinomial"),
           ACC = c(acc_simple, acc_repeat, acc_loocv, acc_multinom),
           F1 = c(f1_simple, f1_repeat, f1_loocv, f1_multinom),
           MCC = c(mcc_simple, mcc_repeat, mcc_loocv, mcc_multinom))
xtable(tabela, type = "latex", digits = c(3,3,3,3,3))
