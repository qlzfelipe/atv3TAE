library(dplyr)

factor_to_dataframe <- function(f) {
  col_names <- levels(f)
  r <- matrix(F, ncol = length(col_names), nrow = length(f), 
              dimnames = list(NULL, col_names)) %>% as.data.frame()
  sapply(1:length(f), FUN = function(x){
    for(i in 1:length(col_names)) {
      if(f[x] == col_names[i]) {
        r[x,i] <<- TRUE; break
      }
    }
  })
  return(r)
}

mcc_custom <- function(actual, predicted) {
  df_actual <- factor_to_dataframe(actual)
  df_predicted <- factor_to_dataframe(predicted)
  
  MCC <- mltools::mcc(preds = df_predicted, actuals = df_actual)
  return(MCC)
}

mcsapply <- function (X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE) {
  FUN <- match.fun(FUN)
  answer <- parallel::mclapply(X = X, FUN = FUN, ...)
  if (USE.NAMES && is.character(X) && is.null(names(answer))) 
    names(answer) <- X
  if (!isFALSE(simplify) && length(answer)) 
    simplify2array(answer, higher = (simplify == "array"))
  else answer
}