lc_bias_gen <- function(data,min_te,max_te){
  gam_res <- gam(I(te_hat-true_te)~s(true_te,k=10),data=data)

  lc_bias <- data.table(
    true_te=seq(min_te,max_te,length=100)
  ) %>%
    .[,bias:=predict(gam_res,newdata=.)]

  bc_mae <- data[,pred_bias:=predict(gam_res,newdata=data)] %>%
    .[,mean(abs(te_hat-true_te-pred_bias))]

  return(list(lc_bias,bc_mae))
}

get_intervals <- function(te_exp,num_int=10){

  set.seed(674384)

  Ng <- 1000000

  data_int <- data.table(
    z1 = runif(Ng),
    z2 = runif(Ng),
    z3 = runif(Ng),
    z4 = runif(Ng)
  )

  #=== define true TE ===#
  eval(parse(text=paste0('data_int[,true_te:=',te_exp,']')))

  min_te <- min(data_int$true_te) - 0.1
  max_te <- max(data_int$true_te) + 0.1
  # te_seq <- seq(min_te,max_te,length=num_int+1)
  # range_labels <- paste(1:num_int)

  return(list(min_te,max_te))
}

data_gen <- function(N,te_exp,di_exp,e_exp,rho,error_sd){

    rho <- 0
    beta <- sqrt(1-rho^2)
    # N <- 10000

    #=== if z and x are correlated ===#
    data <- data.table(
      id = 1:N,
      e = rnorm(N,sd=error_sd),
      u1 = rnorm(N,mean=0.5,sd=1/4),
      u2 = rnorm(N,mean=0.5,sd=1/4),
      u3 = rnorm(N,mean=0.5,sd=1/4),
      u4 = rnorm(N,mean=0.5,sd=1/4),
      D = (rnorm(N,1)/2+1 < 0.5) + 0
    ) %>%
    .[,z1 := beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u1] %>%
    .[,z2 := beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u2] %>%
    .[,z3 := beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u3] %>%
    .[,z4 := beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u4] %>%
    #=== correlated with Z  ===#
    .[,x1:=beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u1] %>%
    .[,x2:=beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u2] %>%
    .[,x3:=beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u3] %>%
    .[,x4:=beta*(rnorm(N,mean=0.5,sd=1/4))+rho*u4]

    #=== define true TE ===#
    eval(parse(text=paste0('data[,true_te:=',te_exp,']')))

    #=== define the direct impacts ===#
    eval(parse(text=paste0('data[,di:=',di_exp,']')))

    #=== error ===#
    eval(parse(text=paste0('data[,e:=',e_exp,'+rnorm(N,sd=error_sd)]')))

    #=== define the dependent variable ===#
    data[,y := di+D*true_te+e]

    return(data)
}

MC_sim <- function(te_exp,di_exp,e_exp,cor,error_sd,cov_var_ls,B,num_cores,type='summary'){

  te_seq <- get_intervals(te_exp)
  min_te <- te_seq[[1]]
  max_te <- te_seq[[2]]

  set.seed(247814)

  #/*=================================================*/
  #' # Define simulation function
  #/*=================================================*/
  sim <- function(i){
    # i <- 1
    #/*----------------------------------*/
    #' ## Data generation
    #/*----------------------------------*/
    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    #' ### Train data
    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    data <- data_gen(N,te_exp,di_exp,e_exp,cor,error_sd)

    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    #' ### test data
    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    data_test <- data_gen(N,te_exp,di_exp,e_exp,cor,error_sd)

    #/*----------------------------------*/
    #' ## Estimation
    #/*----------------------------------*/

    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    #' ### Causal Forest
    #/*~~~~~~~~~~~~~~~~~~~~~~*/
    X <- data[,cov_var_ls,with=FALSE]
    tau_forest <- causal_forest(
      X,
      data[,y],
      data[,D],
      # min.node.size = 20,
      num.trees=4000
    )

    # variable_importance(tau_forest)

    #=== Estimate treatment effects for the test sample ===#
    tau_estimates <- predict(tau_forest, data_test[,cov_var_ls,with=FALSE], estimate.variance = TRUE)

    tau_data_CF <- data_test[,.(id,true_te)] %>%
      .[,te_hat:=tau_estimates$predictions]

    bias_CF <- tau_data_CF[,(te_hat-true_te) %>% mean()]
    mse_CF <- tau_data_CF[,(te_hat-true_te)^2 %>% mean()]
    mae_CF <- tau_data_CF[,abs(te_hat-true_te) %>% mean()]
    lc_res_CF <- lc_bias_gen(tau_data_CF,min_te,max_te)
    local_bias_CF <- lc_res_CF[[1]] %>%
      setnames('bias','bias_CF')
    lbc_mae_CF <- lc_res_CF[[2]]

    # ggplot(data=tau_data_CF) +
    #   geom_point(aes(y=true_te,x=te_hat_CF)) +
    #   geom_abline(intercept=0,slope=1,color='red')

    #/*----------------------------------*/
    #' ## RF
    #/*----------------------------------*/
    #=== causal forest analysis using data ===#
    reg_forest <- regression_forest(
      data[,c(cov_var_ls,'D'),with=FALSE],
      data[,y],
      honesty=TRUE,
      honesty.prune.leaves = TRUE,
      tune.parameters = 'all'
    )

    #=== testing using data_test ===#
    dt_test_0 <- copy(data_test)[,c(cov_var_ls,'D'),with=FALSE] %>%
        .[,D:=0]

    dt_test_1 <- copy(data_test)[,c(cov_var_ls,'D'),with=FALSE] %>%
        .[,D:=1]

    tau_data_RF <- data_test[,.(id,true_te,D,y,e)] %>%
      .[,yhat_0:=predict(reg_forest,dt_test_0)] %>%
      .[,yhat_1:=predict(reg_forest,dt_test_1)] %>%
      .[,te_hat:=yhat_1-yhat_0] %>%
      .[,y_hat_RF:=ifelse(D==1,yhat_1,yhat_0)]

    rsq_RF <- lm(y~y_hat_RF,data=tau_data_RF) %>% summary() %>% .$r.squared

    bias_RF <- tau_data_RF[,(te_hat-true_te) %>% mean()]
    mse_RF <- tau_data_RF[,(te_hat-true_te)^2 %>% mean()]
    mae_RF <- tau_data_RF[,abs(te_hat-true_te) %>% mean()]
    lc_res_RF <- lc_bias_gen(tau_data_RF,min_te,max_te)
    local_bias_RF <- lc_res_RF[[1]] %>%
      setnames('bias','bias_RF')
    lbc_mae_RF <- lc_res_RF[[2]]


    #/*----------------------------------*/
    #' ## XGB
    #/*----------------------------------*/
    dt_train_XGB <- xgb.DMatrix(
      data=data[,c(cov_var_ls,'D'),with=FALSE] %>%
        as.matrix(),
      label=data[,y]
    )

    #=== default parameters ===#
    params <- list(
      booster = "gbtree",
      objective = "reg:squarederror",
      #=== learning rate ===#
      eta = 0.1,
      #=== regularization parameter ===#
      gamma = 1,
      max_depth = 10,
      min_child_weight = 1,
      subsample = 0.8,
      colsample_bytree = 1
    )

    xgbcv <- xgb.cv(
      params = params,
      data = dt_train_XGB,
      nrounds = 100,
      nfold = 10,
      showsd = T,
      stratified = F,
      print_every_n = 100,
      early_stop_round = 20,
      maximize = F
    )

    best_it <- xgbcv$evaluation_log %>%
      .[test_rmse_mean==min(test_rmse_mean),iter]

    xgb_trained <- xgb.train(
      params = params,
      data = dt_train_XGB,
      nrounds = best_it,
      # watchlist = list(val=dt_test,train=dt_train),
      print_every_n = 10,
      early_stop_round = 10,
      maximize = F
    )

    #=== testing using data_test ===#
    dt_test_0 <- xgb.DMatrix(
      data=copy(data_test)[,c(cov_var_ls,'D'),with=FALSE] %>%
        .[,D:=0] %>%
        as.matrix(),
      label=data_test[,y]
    )

    dt_test_1 <- xgb.DMatrix(
      data=copy(data_test)[,c(cov_var_ls,'D'),with=FALSE] %>%
        .[,D:=1] %>%
        as.matrix(),
      label=data_test[,y]
    )

    tau_data_XGB <- data_test[,.(id,true_te,y,D)] %>%
      .[,yhat_0:=predict(xgb_trained,dt_test_0)] %>%
      .[,yhat_1:=predict(xgb_trained,dt_test_1)] %>%
      .[,te_hat:=yhat_1-yhat_0] %>%
      .[,y_hat_XGB:=ifelse(D==1,yhat_1,yhat_0)]

    rsq_XGB <- lm(y~y_hat_XGB,data=tau_data_XGB) %>% summary() %>% .$r.squared

    bias_XGB <- tau_data_XGB[,(te_hat-true_te) %>% mean()]
    mse_XGB <- tau_data_XGB[,(te_hat-true_te)^2 %>% mean()]
    mae_XGB <- tau_data_XGB[,abs(te_hat-true_te) %>% mean()]
    lc_res_XGB <- lc_bias_gen(tau_data_XGB,min_te,max_te)
    local_bias_XGB <- lc_res_XGB[[1]] %>%
      setnames('bias','bias_XGB')
    lbc_mae_XGB <- lc_res_XGB[[2]]

    # ggplot(data=tau_data_XGB) +
    #   geom_point(aes(y=te_hat_XGB,x=true_te)) +
    #   geom_abline(intercept=0,slope=1,color='red')

    # tau_data_CF[tau_data_XGB,on='id'] %>%
    #   .[,.(id,true_te,te_hat_CF,te_hat_XGB)] %>%
    #   melt(id.var=c('id','true_te')) %>%
    #   .[,variable:=cmae_when(
    #     variable == 'te_hat_CF' ~ "Causal Forest",
    #     variable == 'te_hat_XGB' ~ "XGB"
    #   )] %>%
    # ggplot(data=.) +
    #   geom_point(aes(y=value,x=true_te)) +
    #   facet_grid(variable~.) +
    #   geom_abline(intercept=0,slope=1,color='red')

    local_bias_all <- local_bias_CF %>%
      local_bias_RF[.,on='true_te'] %>%
      local_bias_XGB[.,on='true_te'] %>%
      .[,sim:=i]

    if (type=='summary'){
      data_return <- data.table(
        bias_CF = bias_CF,
        bias_RF = bias_RF,
        bias_XGB = bias_XGB,
        mse_CF = mse_CF,
        mse_RF = mse_RF,
        mse_XGB = mse_XGB,
        mae_CF = mae_CF,
        mae_RF = mae_RF,
        mae_XGB = mae_XGB,
        lbc_mae_CF = lbc_mae_CF,
        lbc_mae_RF = lbc_mae_RF,
        lbc_mae_XGB = lbc_mae_XGB,
        rsq_RF = rsq_RF,
        rsq_XGB = rsq_XGB
      )

      return(list(data_return,local_bias_all))

    } else{
      setnames(tau_data_CF,'te_hat','te_hat_CF')
      setnames(tau_data_RF,'te_hat','te_hat_RF')
      setnames(tau_data_XGB,'te_hat','te_hat_XGB')

      data_return <- tau_data_CF %>%
        tau_data_RF[,.(id,te_hat_RF)][.,on='id'] %>%
        tau_data_XGB[,.(id,te_hat_XGB)][.,on='id'] %>%
        .[,sim:=i] %>%
        melt(id.var=c('sim','id','true_te'))

      return(data_return)
    }

  } # end od sim function

  #/*----------------------------------*/
  #' ## Run Simulation
  #/*----------------------------------*/
  #=== simulations ===#
  sim_results <- pbmclapply(
    1:B,
    function(x) sim(x),
    mc.cores=num_cores
  )
  # sim_results <- lapply(
  #   1:B,
  #   function(x) sim(x)
  # )

  #=== put together ===#
  if (type=='summary'){
    return(
      list(
      sapply(sim_results,'[',1) %>%
        rbindlist(),
      sapply(sim_results,'[',2) %>%
        rbindlist()
      )
    )
  } else{
    return(rbindlist(sim_results))
  }

} # end of MC_sim function




