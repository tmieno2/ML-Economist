#' ---
#' title: "MC simluations on the performance of CF and XGB"
#' author: "Taro Mieno"
#' output:
#'   html_document:
#'     number_sections: yes
#'     theme: flatly
#'     highlight: zenburn
#'     toc_float: yes
#'     toc: yes
#'     toc_depth: 3
#' geometry: margin=1in
#' ---

#+ setup, include=FALSE
knitr::opts_chunk$set(
  echo = TRUE,
  cache = FALSE,
  comment = NA,
  message = FALSE,
  warning = FALSE,
  tidy = FALSE,
  cache.lazy = FALSE,
  #--- figure ---#
  dpi=400,
  fig.width=7.5,
  fig.height=5,
  out.width="750px",
  out.height="500px"
)

opts_knit$set(
  root.dir = ""
)

#/*=================================================*/
#' # Preparation
#/*=================================================*/
setwd('~/Box/ResearchProjects/CausalForests_PA_Method')
# setwd('~/Dropbox/CollaborativeResearch/CausalForest_PA_VMs')

library(sf)
library(grf)
library(data.table)
library(parallel)
library(dplyr)
library(ggplot2)
library(magrittr)
library(stringr)
library(mgcv)
library(xgboost)
library(pbmcapply)

#=== source the function ===#
source('./Codes/MC_function.R')

#/*=================================================*/
#' # Objectives
#/*=================================================*/
#' greater portion of non-treatment effects part
#' inclusion of irrelevant variabls
#' error size

#/*=================================================*/
#' # Findings
#/*=================================================*/
#' F2: localized bias for CF, as N goes higher, localized bias reduces
#' bias by quantiles

#' F3: the impact of direct impacts: RF suffers quite a bit. Whether the direct imapct is correlated
#' with the hetrogenous effects matter. Direction of the correlation matters too.

#' F4: correlation of the error term and variable that contribute to hetrogeneous reatment effects

#/*=================================================*/
#' # Simulation
#/*=================================================*/
num_cores <- 6

#/*----------------------------------*/
#' ## The base case
#/*----------------------------------*/
set.seed(7823782)

N <- 1000
B <- 500
te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
di_exp <- '0'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4')
error_sd <- 1

results <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

saveRDS(results,'./Results/base_results.rds')

#/*=================================================*/
#' # Question 1: Localized bias
#/*=================================================*/
set.seed(7823782)
B <- 500

#/*----------------------------------*/
#' ## Model 1
#/*----------------------------------*/
te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
di_exp <- '0'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4')
error_sd <- 1

N <- 1000
results_1000_1 <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)[[2]] %>%
  .[,model:='model 1'] %>%
  .[,N:=1000]

# ggplot(data=results_1000_1[variable=='te_hat_CF' & sim==5,]) +
#   geom_point(aes(y=true_te,x=as.numeric(value))) +
#   geom_abline(slope=1,color='red') +
#   coord_equal()

N <- 5000
results_5000_1 <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)[[2]] %>%
  .[,model:='model 1'] %>%
  .[,N:=5000]

saveRDS(rbind(results_1000_1,results_5000_1),'./Results/local_bias_others.rds')

#/*----------------------------------*/
#' ## Other models
#/*----------------------------------*/
N <- 1000
B <- 500

#=== model 1 ===#
te_exp <- '5*log(z1+2)'
results_1 <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)[[2]] %>%
  .[,model:='model 1']

#=== model 2 ===#
te_exp <- 'pmin(1/pmax(z1,0.2),4)'
results_2 <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)[[2]] %>%
  .[,model:='model 2']

#=== model 3 ===#
te_exp <- '10*(z1-1)^2'
results_3 <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)[[2]] %>%
  .[,model:='model 3']

#=== combine and save ===#
all_results <- rbind(results_1,results_2,results_3)

saveRDS(all_results,'./Results/local_bias_others.rds')


#/*=================================================*/
#' # Question 2: the impact of direct impacts
#/*=================================================*/
#/*----------------------------------*/
#' ## sensitivity to the magnitude of direct impact
#/*----------------------------------*/
#' x5 through x8 uncorrelated with x1 through x4
#'
N <- 1000

te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4')
error_sd <- 1

set.seed(7823782)

alpha_ls <- c(1,3,5,10)
res_ls <- list()

for (alpha in alpha_ls){
  di_exp <- 'alpha*(x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25))'
  res_ls[[paste(alpha)]] <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)
  res_ls[[paste(alpha)]][[1]][,alpha:=alpha]
  res_ls[[paste(alpha)]][[2]][,alpha:=alpha]
}

di_influence <- list(
  sapply(res_ls,'[',1) %>% rbindlist(),
  sapply(res_ls,'[',2) %>% rbindlist()
)

saveRDS(di_influence,'./Results/di_influence.rds')

#/*----------------------------------*/
#' ## when we can ignore $X$
#/*----------------------------------*/
N <- 1000

te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
di_exp <- '5*(x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25))'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4')
error_sd <- 1

di_ignore_vars <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

saveRDS(di_ignore_vars,'./Results/di_ignore_vars.rds')

#/*----------------------------------*/
#' ## when the direct impacts when they are correlated with tretment effects
#/*----------------------------------*/
#' x1 through x4 have direct impacts
#'

# mc.cores <- 4
N <- 1000
B <- 8

te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4')
error_sd <- 1

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### positive correlation
#/*~~~~~~~~~~~~~~~~~~~~~~*/
di_exp <- '5*(x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25))'
res_pos_cor <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

res_pos_cor[[1]][,type_cor:='positive']
res_pos_cor[[2]][,type_cor:='positive']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### negative correlation
#/*~~~~~~~~~~~~~~~~~~~~~~*/
di_exp <- '-5*(x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25))'
res_neg_cor <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

res_neg_cor[[1]][,type_cor:='negative']
res_neg_cor[[2]][,type_cor:='negative']

saveRDS(
  list(
    rbind(res_pos_cor[[1]],res_neg_cor[[1]]),
    rbind(res_pos_cor[[2]],res_neg_cor[[2]])
  ),
  './Results/dir_cor.rds'
)

#/*=================================================*/
#' # Question 3: Correlation of the error term and variables that affect treatment effects
#/*=================================================*/
set.seed(7823782)

N <- 1000
B <- 1000

te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
# di_exp <- '0'
di_exp <- 'x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25)'
rho <- 0
e_exp <- 'ir1+ir2+ir3+ir4'
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4')
error_sd <- 1

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### no correlation
#/*~~~~~~~~~~~~~~~~~~~~~~*/
phi <- 0
results_nc <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

results_nc[[1]][,type:='nc']
results_nc[[2]][,type:='nc']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### correlation
#/*~~~~~~~~~~~~~~~~~~~~~~*/
phi <- 0.9
results_c <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

results_c[[1]][,type:='c']
results_c[[2]][,type:='c']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Combine and save
#/*~~~~~~~~~~~~~~~~~~~~~~*/
all_results_1 <- rbind(results_nc[[1]],results_c[[1]])
all_results_2 <- rbind(results_nc[[2]],results_c[[2]])
saveRDS(list(all_results_1,all_results_2),'./Results/bias_ecor.rds')

#/*=================================================*/
#' # Question 4: Irrelevant variables
#/*=================================================*/
set.seed(7823782)

N <- 1000
B <- 500

te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
# di_exp <- '0'
di_exp <- 'x1^2 + log(x2+1) + x3^3 + 2*x1*x2 + x3/max(x4,0.25)'
rho <- 0
phi <- 0
e_exp <- '0'
error_sd <- 1

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### no irrelevant variables
#/*~~~~~~~~~~~~~~~~~~~~~~*/
phi <- 0
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4')
results_no_ir <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

results_no_ir[[1]][,type:='no_ir']
results_no_ir[[2]][,type:='no_ir']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### non-correlated irrelevant variables
#/*~~~~~~~~~~~~~~~~~~~~~~*/
phi <- 0
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4','ir1','ir2','ir3','ir4')
results_nc_ir <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

results_nc_ir[[1]][,type:='nc_ir']
results_nc_ir[[2]][,type:='nc_ir']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### correlated irrelevant variables
#/*~~~~~~~~~~~~~~~~~~~~~~*/
phi <- 0.9
cov_var_ls <- c('z1','z2','z3','z4','x1','x2','x3','x4','ir1','ir2','ir3','ir4')
results_c_ir <- MC_sim(te_exp,di_exp,e_exp,rho,phi,error_sd,cov_var_ls,B,num_cores)

results_c_ir[[1]][,type:='c_ir']
results_c_ir[[2]][,type:='c_ir']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Combine and save
#/*~~~~~~~~~~~~~~~~~~~~~~*/
all_results_1 <- rbind(results_no_ir[[1]],results_nc_ir[[1]],results_c_ir[[1]])
all_results_2 <- rbind(results_no_ir[[2]],results_nc_ir[[2]],results_c_ir[[2]])

saveRDS(list(all_results_1,all_results_2),'./Results/irl_var_results.rds')

#/*=================================================*/
#' # Sim for visualization
#/*=================================================*/
#/*----------------------------------*/
#' ## localized bias
#/*----------------------------------*/
set.seed(7823782)

N <- 1000
B <- 4
di_exp <- '0'
rho <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4')
error_sd <- 1

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### The base case
#/*~~~~~~~~~~~~~~~~~~~~~~*/
te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
results_b <- MC_sim(te_exp,di_exp,e_exp,rho,error_sd,cov_var_ls,B,num_cores,type='detailed') %>%
  .[,type:='base']

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### a simpler case
#/*~~~~~~~~~~~~~~~~~~~~~~*/
te_exp <- '5*log(z1+2)'
results_s <- MC_sim(te_exp,di_exp,e_exp,rho,error_sd,cov_var_ls,B,num_cores,type='detailed') %>%
  .[,type:='simple']

saveRDS(rbind(results_b,results_s),'./Results/lbc_results_viz.rds')

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### 95% interval
#/*~~~~~~~~~~~~~~~~~~~~~~*/
N <- 1000000
di_exp <- '0'
rho <- 0
phi <- 0
e_exp <- '0'
cov_var_ls <- c('z1','z2','z3','z4')
error_sd <- 1

#=== model 1 ===#
te_exp <- '2+z1^2 + sqrt(z1+z2+1) + (z2+z3)^2 - 1/pmax(z1*z4,0.5) + z4'
interval_95_model_1 <- data_gen(N,te_exp,di_exp,e_exp,rho,phi,error_sd) %>%
  .[,quantile(true_te,prob=c(0.025,0.125,0.875,0.975))]

#=== model 1 ===#
te_exp <- '5*log(z2+2)'
interval_95_model_2 <- data_gen(N,te_exp,di_exp,e_exp,rho,phi,error_sd) %>%
  .[,quantile(true_te,prob=c(0.025,0.125,0.875,0.975))]

saveRDS(list(interval_95_model_1,interval_95_model_2),'./Results/intervals_95.rds')


