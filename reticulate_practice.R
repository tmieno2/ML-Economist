# /*----------------------------------*/
#' ## Economic parameters
# /*----------------------------------*/
price_table <- data.table(
  # === $/kg corn ===#
  pCorn = round(c(3.5, 3.5, 5) * 200 / 5080, digits = 3),
  # === $/kg N ===#
  pN = round(c(0.4, 0.6, 0.4) / 0.453592, digits = 3)
)


# /*=================================================*/
#' # Run main simulations
# /*=================================================*/
sim_par <- function(i, var_ls, reg_data, test_data, N_levels) {
  print(paste0("working on ", i, " th iteration."))
  # i = 1; var_ls = c("alpha", "beta", "ymax")
  #' ## run ML analyses

  case_data <- 
    tibble(
      Method = c("CF_base", "XGBRF", "RF_split", "DML_base", "CF_DML", "X_Learner", "S_Learner", "T_Learner", "Doubly_Robust")    
    ) 

  results_data <- 
    case_data %>%
      mutate(
        opt_N_data = lapply(              
          seq_len(nrow(.)),
          function(x) {
            ### === apply various methods to get optimal N ===###
            get_opt_N(
              reg_data = reg_data,
              test_data = test_data,
              var_ls = var_ls,
              rates_ls = N_levels,
              Method = .$Method[[x]]
            )
          }
        # , future.seed = TRUE
      )
    )%>%
    unnest(., cols= "opt_N_data")%>%
    data.table()

  return(results_data)
     
}

get_opt_N <- function(reg_data, test_data, var_ls, rates_ls, Method) {
  eval_dt <- bind_rows(reg_data, test_data, .id = "type") %>%
    .[, type:= case_when(type == 1 ~ "train", type == 2 ~ "test")]
    ### === CF_base ===###
  if (Method == "CF_base") {
    opt_N_data <-
      CF_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_base(
        test_data = eval_dt, 
        cf_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
    .[, pred_yield := NA]
    ### === Double ML ===###
  } else if (Method == "DML_base") {
    opt_N_data <- 
      DML_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === CF_DML ===###
  } else if (Method == "CF_DML") {
    opt_N_data <- 
      CF_DML_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === X_Learner ===###
  } else if (Method == "X_Learner") {
    opt_N_data <- 
      X_Learner_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === S_Learner ===###
  } else if (Method == "S_Learner") {
    opt_N_data <- 
      S_Learner_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === T_Learner ===###
  } else if (Method == "T_Learner") {
    opt_N_data <- 
      T_Learner_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === Doubly_Robust ===###
  } else if (Method == "Doubly_Robust") {
    opt_N_data <- 
      Doubly_Robust_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      )%>%
      get_pi_dif_dml(
        test_data = eval_dt, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )%>%
      .[, pred_yield := NA]
    ### === XGBRF ===###
  } else if (Method == "XGBRF") {
      opt_N_data <- 
        XGBRF_run(
          reg_data = reg_data,
          var_ls = var_ls
        ) %>%
        XGBRF_analysis(
          test_data = eval_dt,
          f_results = .,
          var_ls = var_ls,
          N_levels = rates_ls
        )
      ### === RF_split ===###
  } else if (Method == "RF_split") {
      opt_N_data <- 
        RF_split_run(
          reg_data = reg_data,
          var_ls = var_ls
        ) %>%
        RF_split_analysis(
          test_data = eval_dt,
          f_results = .,
          var_ls = var_ls,
          N_levels = rates_ls
        )
   }

  # === arrange order of columns === #  
  report_data <- opt_N_data[,.(unique_subplot_id, type, opt_N_hat, pred_yield, sim, yield, opt_N)]
  return(report_data)
}

# /**==================================================**/
#' 1-2, 1-3, 1-4, 1-5 CF treatment effect estimation ("base")
# /**==================================================**/
CF_analysis_base <- function(reg_data, var_ls) {

  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1

  CF_all_run <- function(i) {
    tau_forest_temp <- CF_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from CF_analysis
      var_ls = var_ls
    )

    return(tau_forest_temp)
  }

  all_results <- lapply(1:exp_len, CF_all_run)

  return(all_results)
}
# /*================================================================*/
#' # Run CF
# /*================================================================*/
CF_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === preliminary runs ===#
  Y_forest <- regression_forest(X, Y)
  Y_hat <- predict(Y_forest)$predictions
  
  W_forest <- regression_forest(X, W)
  W_hat <- predict(W_forest)$predictions
  
  # === CF analysis ===#
  tau_forest_temp <- causal_forest(X, Y, W,
                                   Y.hat = Y_hat,
                                   W.hat = W_hat,
                                   honesty = TRUE,
                                   num.trees = 2000,
                                   num.threads = 1,
                                   tune.parameters = "all"
  )
  return(tau_forest_temp)
}
get_pi_dif_base <- function(test_data, cf_results, var_ls, rates_ls) {
  # test_data=eval_dt; cf_results=bcf_base
  # var_ls=var_ls; rates_ls=N_levels
  
  # === Prediction of Treatment Effects === #  
  pi_dif_data <- lapply(
    1:(length(cf_results) + 1),
    function(x) {
      get_changes_gradual_base(
        x,
        data_base = test_data,
        var_ls = var_ls,
        rates_ls = rates_ls,
        cf_results = cf_results
      )
    }
  ) %>%
    rbindlist() %>%
    # === estimate EONR === #
    .[, pi_change := pCorn * yield_dif - pN * N_plus] %>%
    .[, .SD[pi_change == max(pi_change), ], by = .(unique_subplot_id, type)] %>%
    .[, .(unique_subplot_id, type, N)] %>%
    setnames("N", "opt_N_hat") %>%
    .[test_data[,.(type, sim, yield, opt_N, unique_subplot_id)], on = c("unique_subplot_id", "type")] %>%
    .[, pred_yield := NA]
  
  return(pi_dif_data)
}
get_changes_gradual_base <- function(N_index, data_base, var_ls, rates_ls, cf_results) {
  # data_base=eval_dt; var_ls=var_ls; rates_ls=N_levels; cf_results=cf_base
  # /*----------------------------------*/
  #' ## Increase N
  # /*----------------------------------*/
  # N_index=5
  for (i in 1:N_index) {
    data_eval_base <- data_base %>%
      .[, ..var_ls]
    
    if (i == 1) {
      tau_data <- rep(0, nrow(data_base)) # <- Actually, this part is unnecessary, but just for the purpose of consistency
    } else {
      tau_data <- predict(cf_results[[i - 1]], newdata = data_eval_base, estimate.variance = FALSE)$predictions
    }
  }
  
  return_data <- data.table(
    unique_subplot_id = data_base$unique_subplot_id,
    type = data_base$type,
    yield_dif = tau_data, # yield diff shows the treatment effects
    N_plus = rates_ls[N_index] - rates_ls[1],
    N = rates_ls[N_index]
  )
  
  return(return_data)
}



# /**==================================================**/
#' 1-2, 1-3, 1-4, 1-5 DML treatment effect estimation ("base")
# /**==================================================**/
DML_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  CF_all_run <- function(i) {
    tau_forest_temp <- DML_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, CF_all_run)
  
  return(all_results)
}

DML_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
   # === CF analysis ===#
  tau_forest_temp <- estDML$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}
get_pi_dif_dml <- function(test_data, dml_results, var_ls, rates_ls) {
  # test_data=eval_dt; cf_results=bcf_base
  # var_ls=var_ls; rates_ls=N_levels
  
  # === Prediction of Treatment Effects === #  
  pi_dif_data <- lapply(
    1:(length(dml_results) + 1),
    function(x) {
      get_changes_gradual_dml(
        x,
        data_base = test_data,
        var_ls = var_ls,
        rates_ls = rates_ls,
        dml_results = dml_results
      )
    }
  ) %>%
    rbindlist() %>%
    # === estimate EONR === #
    .[, pi_change := pCorn * yield_dif - pN * N_plus] %>%
    .[, .SD[pi_change == max(pi_change), ], by = .(unique_subplot_id, type)] %>%
    .[, .(unique_subplot_id, type, N)] %>%
    setnames("N", "opt_N_hat") %>%
    .[test_data[,.(type, sim, yield, opt_N, unique_subplot_id)], on = c("unique_subplot_id", "type")] %>%
    .[, pred_yield := NA]
  
  return(pi_dif_data)
}
get_changes_gradual_dml <- function(N_index, data_base, var_ls, rates_ls, dml_results) {
  # data_base=eval_dt; var_ls=var_ls; rates_ls=N_levels; cf_results=cf_base
  # /*----------------------------------*/
  #' ## Increase N
  # /*----------------------------------*/
  # N_index=5
  data_eval_base <- data_base %>%
      .[, ..var_ls]
    
    if (N_index == 1) {
      tau_data <- rep(0, nrow(data_base)) # <- Actually, this part is unnecessary, but just for the purpose of consistency
    } else {
      tau_data <- dml_results[[N_index - 1]]$effect(data_eval_base)
    }

  return_data <- data.table(
    unique_subplot_id = data_base$unique_subplot_id,
    type = data_base$type,
    yield_dif = tau_data, # yield diff shows the treatment effects
    N_plus = rates_ls[N_index] - rates_ls[1],
    N = rates_ls[N_index]
  )
  
  return(return_data)
}



# /**==================================================**/
#' 1-2, 1-3, 1-4, 1-5 Python EconML treatment effect estimation ("base")
# /**==================================================**/
CF_DML_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  CF_DML_all_run <- function(i) {
    tau_forest_temp <- CF_DML_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, CF_DML_all_run)
  
  return(all_results)
}
CF_DML_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === CF analysis ===#
  tau_forest_temp <- estCF$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}

X_Learner_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  X_Learner_all_run <- function(i) {
    tau_forest_temp <- X_Learner_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, X_Learner_all_run)
  
  return(all_results)
}
X_Learner_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === CF analysis ===#
  tau_forest_temp <- estXL$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}

S_Learner_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  S_Learner_all_run <- function(i) {
    tau_forest_temp <- S_Learner_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, S_Learner_all_run)
  
  return(all_results)
}
S_Learner_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === CF analysis ===#
  tau_forest_temp <- estSL$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}

T_Learner_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  T_Learner_all_run <- function(i) {
    tau_forest_temp <- T_Learner_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, T_Learner_all_run)
  
  return(all_results)
}
T_Learner_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === CF analysis ===#
  tau_forest_temp <- estTL$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}

Doubly_Robust_analysis_base <- function(reg_data, var_ls) {
  
  rates_ls <- reg_data[, rate] %>%
    unique() %>%
    sort()
  exp_len <- length(rates_ls) - 1
  
  Doubly_Robust_all_run <- function(i) {
    tau_forest_temp <- Doubly_Robust_run(
      data = reg_data,
      rates = rates_ls[c(1, i + 1)], # <- this part is different from DML_analysis
      var_ls = var_ls
    )
    
    return(tau_forest_temp)
  }
  
  all_results <- lapply(1:exp_len, Doubly_Robust_all_run)
  
  return(all_results)
}
Doubly_Robust_run <- function(data, rates, var_ls) {
  
  # === treatment assignment ===#
  data_temp_dt <- data %>%
    .[rate %in% rates, ] %>%
    .[, trt := ifelse(rate == rates[1], 0, 1)]
  
  # === Preparation ===#
  X <- data_temp_dt[, ..var_ls]
  Y <- data_temp_dt[, yield]
  W <- data_temp_dt[, trt]
  
  # === CF analysis ===#
  tau_forest_temp <- estDRnon$fit(Y, W, X=X)
  
  return(tau_forest_temp)
}





# /*================================================================*/
#' # Run XGBRF, RF_split
# /*================================================================*/

#/*----------------------------------*/
#' ## Run XGBRF
#/*----------------------------------*/
XGBRF_run <- function(reg_data, var_ls) {
  
  # === Preparation === #
  X <- reg_data[, c("aa_n", var_ls), with = FALSE] 
  Y <- reg_data[, yield]
  
  # === causal forest analysis ===#
  XGBRF_temp <- xgboost(
    data= as.matrix(X),
    label= Y,
    max.depth= 4,
    eta= 0.1,
    nthread=1,
    nrounds= 200,
    objective= "reg:squarederror",
    verbose = 0
  )
  return(XGBRF_temp)
}


#/*----------------------------------*/
#' ## Run RF_split
#/*----------------------------------*/
RF_split_run <- function(reg_data, var_ls) {
  
  # === Preparation === #
  X <- as.formula(paste0("yield~aa_n +",paste0(var_ls, collapse = "+")))

  # === causal forest analysis ===#
  RF_split_temp <- ranger(
    formula = X,
    data= reg_data,
    num.trees = 2000, 
    importance= "impurity",
    split.select.weights= c(1,rep(0.1, length(var_ls)) )
  )
  return(RF_split_temp)
}


#/*----------------------------------*/
#' ## XGBRF analysis 
#/*----------------------------------*/
XGBRF_analysis <- function(test_data, f_results, var_ls, N_levels) {
  
  N_seq <- seq(min(N_levels), max(N_levels), by = 1)
  
  result <- 
    test_data %>%
    .[,c("sim", "unique_subplot_id", "type","aa_n", var_ls, "opt_N", "yield"), with=FALSE] %>%
    # === Yield Prediction === #
    .[, pred_yield := predict(f_results, newdata = as.matrix(.[, c("aa_n", var_ls), with = FALSE]))] %>%
    # === EONR estimation === #
    .[rep(1:nrow(.), each = length(N_seq)),] %>%
    .[, rate := rep(N_seq, nrow(.) / length(N_seq))] %>%
    .[, yield_hat := predict(f_results, newdata = .[, c("rate", var_ls), with = FALSE] %>% 
                             setnames("rate", "aa_n")%>% as.matrix())] %>%
    .[, pi_hat := pCorn * yield_hat - pN * rate]%>%
    .[, .SD[pi_hat == max(pi_hat), ], by = .(unique_subplot_id, type)] %>%
    setnames("rate", "opt_N_hat") %>%
    .[, .(unique_subplot_id, type, opt_N_hat, pred_yield, sim, yield, opt_N)]
  return(result)
}


#/*----------------------------------*/
#' ## RF_split analysis 
#/*----------------------------------*/
RF_split_analysis <- function(test_data, f_results, var_ls, N_levels) {
  
  N_seq <- seq(min(N_levels), max(N_levels), by = 1)
  
  result <- 
    test_data %>% 
    .[,c("sim", "unique_subplot_id", "type","aa_n", var_ls, "opt_N", "yield"), with=FALSE] %>%
    # === Yield Prediction === #
    .[, pred_yield := predict(f_results, data = .[, c("aa_n", var_ls), with = FALSE])$predictions] %>%
    # === EONR estimation === #
    .[rep(1:nrow(.), each = length(N_seq)),] %>%
    .[, rate := rep(N_seq, nrow(.) / length(N_seq))] %>%
    .[, yield_hat := predict(f_results, data = .[, c("rate", var_ls), with = FALSE] %>% setnames("rate", "aa_n"))$predictions] %>%
    .[, pi_hat := pCorn * yield_hat - pN * rate]%>%
    .[, .SD[pi_hat == max(pi_hat), ], by = .(unique_subplot_id, type)] %>%
    setnames("rate", "opt_N_hat") %>%
    .[, .(unique_subplot_id, type, opt_N_hat, pred_yield, sim, yield, opt_N)]
  return(result)
}













# /* 3_2_CompTeEstimation
# /*================================================================*/
#' # Treatment effect comparison (CF-base vs RF vs BRF) 
# /*================================================================*/
get_te_dt <- function(reg_data, test_data, var_ls, rates_ls, Method) {

    ### === CF_base ===###
  if (Method == "CF_base") {
    te_data <- 
      CF_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      ) %>%
      CF_base_calculate_te(
        test_data = test_data, 
        cf_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )
    ### === DML ===###
  } else if (Method == "DML_base") {
    te_data <- 
      DML_analysis_base(
        reg_data = reg_data,
        var_ls = var_ls
      ) %>%
      DML_base_calculate_te(
        test_data = test_data, 
        dml_results= .,
        var_ls = var_ls,
        rates_ls = rates_ls
      )
    ### === BRF ===###
  } else if (Method == "BRF") {
    te_data <- 
      BRF_run(
        reg_data = reg_data,
        var_ls = var_ls
      ) %>%
      RF_BRF_calculate_te(
        test_data = test_data,
        f_results = .,
        var_ls = var_ls,
        N_levels = rates_ls
      )
    ### === RF ===###
  } else if (Method == "RF") {
    te_data <- 
      RF_run(
        reg_data = reg_data,
        var_ls = var_ls
      ) %>%
      RF_BRF_calculate_te(
        test_data = test_data,
        f_results = .,
        var_ls = var_ls,
        N_levels = rates_ls
      )
    ### === XGBRF ===###
  } else if (Method == "XGBRF") {
    te_data <- 
      XGBRF_run(
        reg_data = reg_data,
        var_ls = var_ls
      ) %>%
      XGBRF_calculate_te(
        test_data = test_data,
        f_results = .,
        var_ls = var_ls,
        N_levels = rates_ls
      )
  } else if (Method == "RF_split") {
      te_data <- 
        RF_split_run(
          reg_data = reg_data,
          var_ls = var_ls
        ) %>%
        RF_split_calculate_te(
          test_data = test_data,
          f_results = .,
          var_ls = var_ls,
          N_levels = rates_ls
        )
  }
  
  return(te_data)
}


#/*----------------------------------*/
#' ## CF-base
#/*----------------------------------*/
CF_base_calculate_te <- function(test_data, cf_results, var_ls, rates_ls) {
  # test_data=test_dt; cf_results=cf_base
  # var_ls = c("alpha","beta","ymax"); rates_ls = N_levels

  te_data <- lapply(
    1:(length(cf_results) + 1),
    function(x) {
      get_changes_gradual_base(
        x,
        data_base = test_data,
        var_ls = var_ls,
        rates_ls = rates_ls,
        cf_results = cf_results
      )
    }
  ) %>%
  rbindlist() %>%
  setnames("N", "rate") %>%
  setnames("yield_dif", "te_base") %>%
  .[,.(unique_subplot_id, rate, te_base)]

  return(te_data)
}


#/*----------------------------------*/
#' ## DML-base
#/*----------------------------------*/
DML_base_calculate_te <- function(test_data, dml_results, var_ls, rates_ls) {
  # test_data=test_dt; cf_results=cf_base
  # var_ls = c("alpha","beta","ymax"); rates_ls = N_levels
  
  te_data <- lapply(
    1:(length(dml_results) + 1),
    function(x) {
      get_changes_gradual_dml(
        x,
        data_base = test_data,
        var_ls = var_ls,
        rates_ls = rates_ls,
        dml_results = dml_results
      )
    }
  ) %>%
    rbindlist() %>%
    setnames("N", "rate") %>%
    setnames("yield_dif", "te_base") %>%
    .[,.(unique_subplot_id, rate, te_base)]
  
  return(te_data)
}


#/*----------------------------------*/
#' ## XGBRF
#/*----------------------------------*/
XGBRF_calculate_te <- function(test_data, f_results, var_ls, N_levels) {
  te_data <- test_data[, c("unique_subplot_id", var_ls, "aa_n", "opt_N", "yield", "X", "Y"), with = FALSE] %>%
    .[rep(1:nrow(.), each = length(N_levels)), ] %>%
    .[, rate := rep(N_levels, nrow(.) / length(N_levels))] %>%
    .[, yield_hat := predict(f_results, newdata = .[, c("rate", var_ls), with = FALSE] %>%
                              setnames("rate", "aa_n")%>% as.matrix())] %>%
    #--- Treatment effect calculation ---#
    .[, yield_base := .SD[rate==min(rate), yield_hat], by = .(unique_subplot_id)] %>%
    .[, te_base := yield_hat - yield_base] %>%
    .[, .(unique_subplot_id, rate, te_base)]
  
  return(te_data)
}


#/*----------------------------------*/
#' ## RF_split
#/*----------------------------------*/
RF_split_calculate_te <- function(test_data, f_results, var_ls, N_levels) {

  te_data <- test_data[, c("unique_subplot_id", var_ls, "aa_n", "opt_N", "yield", "X", "Y"), with = FALSE] %>%
    .[rep(1:nrow(.), each = length(N_levels)), ] %>%
    .[, rate := rep(N_levels, nrow(.) / length(N_levels))] %>%
    .[, yield_hat := predict(f_results, data = .[, c("rate", var_ls), with = FALSE] %>% setnames("rate","aa_n"))$predictions] %>%
    #--- Treatment effect calculation ---#
    .[, yield_base := .SD[rate==min(rate), yield_hat], by = .(unique_subplot_id)] %>%
    .[, te_base := yield_hat - yield_base] %>%
    .[, .(unique_subplot_id, rate, te_base)]
  
  return(te_data)
}


#/*----------------------------------*/
#' ## BRF and RF
#/*----------------------------------*/
RF_BRF_calculate_te <- function(test_data, f_results, var_ls, N_levels) {
  # test_data = test_data_sample; f_results = temp_BRF
  # var_ls = c("alpha","beta","ymax"); N_levels = N_levels 

  te_data <- test_data[, c("unique_subplot_id", var_ls, "aa_n", "opt_N", "yield", "X", "Y"), with = FALSE] %>%
    .[rep(1:nrow(.), each = length(N_levels)), ] %>%
    .[, rate := rep(N_levels, nrow(.) / length(N_levels))] %>%
    .[, yield_hat := predict(f_results, newdata = .[, c("rate", var_ls), with = FALSE])] %>%
    #--- Treatment effect calculation ---#
    .[, yield_base := .SD[rate==min(rate), yield_hat], by = .(unique_subplot_id)] %>%
    .[, te_base := yield_hat - yield_base] %>%
    .[, .(unique_subplot_id, rate, te_base)]

  return(te_data)
}




