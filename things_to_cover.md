# Syllabus

## Textbook

+ [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/B006UTJ1NC/ref=asc_df_B006UTJ1NC/?tag=hyprod-20&linkCode=df0&hvadid=241983376253&hvpos=&hvnetw=g&hvrand=10888417445404071516&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9024744&hvtargid=pla-584543289341&psc=1)

+ [Introduction to Statistical Learning]

# List of things to cover

## Prediction vs Causal Inference
+ why we do not care about prediction 99% of the time
+ why a good prediction does not necessarily lead to a better causal impact identification

## Use python from within R using the reticulate package

### set up python
### how to use reticulate
https://www.r-bloggers.com/2020/04/how-to-run-pythons-scikit-learn-in-r-in-5-minutes/

## prediction-oriented ML methods
+ [x] LASSO 
+ [x] RIDGE 
+ [x] Bagging
+ [x] Tree
+ [x] Boosted Regression Forest 
+ Neural Networks (NA)
+ Deep Learning (NA)

## Cross-validation for tuning and model selection

(scikitlearn)[https://scikit-learn.org/stable/modules/cross_validation.html]

## Causal inference with Machine Learning

+ R-learner 
+ Double Machine Learning (DML)
+ Doubly-Robust (DR)

+ Orthogonal Forest
+ Causal Forest

## Codes

+ `grf`
  + random forest
  + boosted_regression_forest
  + causal_forest

## Python packages
+ scikitlearn

## Extensions

+ Fixed effects
+ nonlinearity of the impact 


## MC experiments

+ Simple linear regression to introduce Loss function and K-nearest neighbor
+ GAM 
+ MC simulations to check how well CV works in picking the best model
+ Non-parametric model is worse than a parametric model when the parametric model is specified correctly.
+ MC: How does predictive ML perform under endogeneity problem
+ What happens to DML when $\eta$ and $\varepsilon$ are not orthogonal
+ What happen if you over-fit the first stage of DML?

# Resources

+ https://daviddalpiaz.github.io/r4sl/