library(reticulate)
repl_python()

library(data.table)

py$a
dml <- py$est

dml()
py$X_test
dml$fit(
  py$Y_train, 
  py$T_train, 
  X = py$X_train, 
  W = py$W_train
)

te_pred = dml$effect(py$X_test)
