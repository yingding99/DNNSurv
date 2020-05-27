# Loss function for the DNNSurv model, based on the negative Cox partial likelihood function
# Using Efron's approximation to handle tied events
# y_true: n X 2 matrix with columne names "time" and "event", corresponding to observed time and censoring status
# y_pred: n X 1 matrix
# return -loglik
loss_lik_efron <- function(y_true, y_pred) {
  time = y_true[,1]
  event = y_true[,2]
  y_pred = k_flatten(y_pred)
  y_pred = tf$cast(y_pred, tf$float32)
  
  n = tf$shape(time)[1]
  sort_index = tf$nn$top_k(time, k = n, sorted = T)$indices
  time = k_gather(reference = time, indices = sort_index)
  event = k_gather(reference = event, indices = sort_index)
  y_pred = k_gather(reference = y_pred, indices = sort_index)
  
  time = k_reverse(time, axes = 1)
  event = k_reverse(event, axes = 1)
  y_pred = k_reverse(y_pred, axes = 1)
  
  time_event = time * event
  unique_ftime = tf$unique(tf$boolean_mask(tensor = time_event, mask = tf$greater(time_event, 0)))$y
  m = tf$shape(unique_ftime)[1]
  tie_count = tf$unique_with_counts(tf$boolean_mask(time_event, tf$greater(time_event, 0)))$count
  ind_matrix = k_expand_dims(time, 1) - k_expand_dims(time, 2)
  ind_matrix = k_equal(x = ind_matrix, y = k_zeros_like(ind_matrix))
  ind_matrix = k_cast(x = ind_matrix, dtype = tf$float32)
  
  time_count = k_cumsum(tf$unique_with_counts(time)$count)
  time_count = k_cast(time_count - k_ones_like(time_count), dtype = tf$int32)
  ind_matrix = k_gather(ind_matrix, time_count)
  
  tie_haz = k_exp(y_pred) * event
  tie_haz = k_dot(ind_matrix, k_expand_dims(tie_haz))
  event_index = tf$math$not_equal(tie_haz,0)
  tie_haz = tf$boolean_mask(tie_haz, event_index)
  
  tie_risk = y_pred * event
  tie_risk = k_dot(ind_matrix, k_expand_dims(tie_risk))
  tie_risk = tf$boolean_mask(tie_risk, event_index)
  
  cum_haz = k_dot(ind_matrix, k_expand_dims(k_exp(y_pred)))
  cum_haz = k_reverse(tf$cumsum(k_reverse(cum_haz, axes = 1)), axes = 1)
  cum_haz = tf$boolean_mask(cum_haz, event_index)
  
  likelihood = tf$Variable(0., trainable = F) 
  j = tf$cast(0, dtype = tf$int32)
  loop_cond = function(j, ...) {return(j < m)}
  
  loop_body = function(j, tc, tr, th, ch, lik) {
    
    l = tc[j]
    l = k_cast(l, dtype = tf$float32)
    J = tf$linspace(start = tf$cast(0, tf$float32), stop = l-1, num = tf$cast(l, tf$int32))/l 
    Dm = ch[j] - J*th[j]
    
    lik = lik + tr[j] - tf$math$reduce_sum(tf$math$log(Dm))
    
    one = k_ones_like(j)
    j_new = j + one
    return(list(j_new, tc, tr, th, ch, lik))
  }

  loop_out = tf$while_loop(cond = loop_cond, body = loop_body,
                          loop_vars = list(j, tie_count, tie_risk, tie_haz, cum_haz, likelihood))
  log_lik = loop_out[length(loop_out)][[1]]

  return(tf$negative(log_lik))

}



# Obtain estimated baseline cumulative hazards and baseline survival probabilities
base_efron <- function(y_test, y_test_pred) {

  time = y_test[,1]
  event = y_test[,2]
  y_pred = y_test_pred

  n = length(time)
  sort_index = order(time, decreasing = F)
  time = time[sort_index]
  event = event[sort_index]
  y_pred = y_pred[sort_index]

  time_event = time * event
  unique_ftime = unique(time[event!=0])
  m = length(unique_ftime)

  tie_count = as.numeric(table(time[event!=0]))

  ind_matrix = matrix(rep(time, times = length(time)), ncol = length(time)) - t(matrix(rep(time, times = length(time)), ncol = length(time)))
  ind_matrix = (ind_matrix == 0)
  ind_matrix[ind_matrix == TRUE] = 1
  time_count = as.numeric(cumsum(table(time)))
  ind_matrix = ind_matrix[time_count,]

  tie_haz = exp(y_pred) * event
  tie_haz = ind_matrix %*% matrix(tie_haz, ncol = 1)
  event_index = which(tie_haz!=0)
  tie_haz = tie_haz[event_index,]

  cum_haz = (ind_matrix %*% matrix(exp(y_pred), ncol = 1))
  cum_haz = rev(cumsum(rev(cum_haz)))
  cum_haz = cum_haz[event_index]

  base_haz = c()
  j = 1
  while(j < m+1) {
    l = tie_count[j]
    J = seq(from = 0, to = l-1, length.out = l)/l
    Dm = cum_haz[j] - J*tie_haz[j]
    Dm = 1/Dm
    Dm = sum(Dm)

    base_haz = c(base_haz, Dm)
    j = j+1
  }

  base_haz = cumsum(base_haz)

  base_haz_all = unlist(sapply(time, function(x){ifelse(sum(unique_ftime <= x) == 0, 0, base_haz[unique_ftime==max(unique_ftime[which(unique_ftime <= x)])])}), use.names = F)
  if (length(base_haz_all) < length(time)) {
    base_haz_all <- c(rep(0, length(time) - length(base_haz_all)), base_haz_all)
  }

  return(list(cumhazard = unique(data.frame(hazard=base_haz_all, time = time)),
              survival = unique(data.frame(surv=exp(-base_haz_all), time = time))))
}

# Obtain Brier scores
brier_efron <- function(y_train_true, y_train_pred, y_newdata, y_newdata_pred, times){

  baseline <- base_efron(y_train_true, y_train_pred)

  y_newdata <- data.frame(y_newdata)
  colnames(y_newdata) = c("time","event")
  new_index <- order(y_newdata$time)
  y_newdata <- y_newdata[new_index,]
  y_newdata_pred <- y_newdata_pred[new_index,]

  Y_x = sapply(times, function(x){as.integer(y_newdata$time > x)})

  ipcw <- pec::ipcw(formula = as.formula(Surv(time, event) ~ 1),
                    data = y_newdata,
                    method = "marginal",
                    times = times,
                    subjectTimes = y_newdata$time,
                    subjectTimesLag = 1)
  G_t = ipcw$IPCW.times
  G_x = ipcw$IPCW.subjectTimes

  W_x = matrix(NA, nrow = nrow(y_newdata), ncol = length(times))
  for (t in 1:length(times)) {

    W_x[,t] = (1-Y_x[,t])*y_newdata$event/G_x + Y_x[,t]/G_t[t]

  }

  Lambda_t = sapply(times, function(x){baseline$cumhazard$hazard[sum(baseline$cumhazard$time <= x)]  })
  S_x = exp(-1 * exp(y_newdata_pred) %*% matrix(Lambda_t, nrow = 1))

  BS_t = sapply(1:length(times), function(x) {mean(W_x[,x] * (Y_x[,x] - S_x[,x])^2)})

  return(list(bs = data.frame(time=times, bs=BS_t)))
}


# two utility functions for running LIME using the DNN survival model
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  return("regression")
}

predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  newdata <- as.matrix(newdata)
  y_dat_pred <- x %>% predict(newdata)
  return(data.frame(Response = y_dat_pred))
}
