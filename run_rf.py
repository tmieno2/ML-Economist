def train_test_RF(n_samples, n_features, n_informative, noise, rng):

  from sklearn.model_selection import train_test_split
  from sklearn.datasets import make_regression
  from sklearn.ensemble import RandomForestRegressor
  import numpy as np
  
  X, y = make_regression(
    n_samples, 
    n_features, 
    n_informative = n_informative, 
    noise = noise, 
    random_state = rng
  )
  
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, random_state = rng
  )

  reg_rf = RandomForestRegressor(max_features = "sqrt")

  reg_rf.fit(X_train, y_train)

  test_score = reg_rf.score(X_test, y_test)

  return test_score

