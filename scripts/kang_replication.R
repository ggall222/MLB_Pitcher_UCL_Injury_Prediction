# Parent-paper experiment: R XGBoost + mgcv. No UCL/chronological extensions.
suppressPackageStartupMessages({library(xgboost); library(mgcv); library(jsonlite)})

auc <- function(y, score, weight=rep(1,length(y))) {
  o <- order(score); y <- y[o]; weight <- weight[o]; score <- score[o]
  group <- cumsum(c(TRUE,diff(score)!=0))
  pos <- as.numeric(rowsum(weight*y,group,reorder=FALSE))
  neg <- as.numeric(rowsum(weight*(1-y),group,reorder=FALSE))
  if (sum(pos)==0 || sum(neg)==0) return(NA_real_)
  sum(pos*(cumsum(neg)-neg/2))/(sum(pos)*sum(neg))
}

folds <- function(y,k,seed) {
  set.seed(seed); f <- integer(length(y))
  for (v in c(0,1)) { ix <- which(y==v); f[ix] <- sample(rep(seq_len(k),length.out=length(ix))) }
  f
}

stratified_subset <- function(y,fraction,seed) {
  set.seed(seed)
  sort(unlist(lapply(c(0,1),function(v) { ix<-which(y==v); sample(ix,floor(length(ix)*fraction)) })))
}

fit_model <- function(x,y,cfg,seed) {
  if (length(unique(y))!=2) stop('Two outcome classes required')
  counts <- table(y); weights <- length(y)/(2*as.numeric(counts[as.character(y)]))
  d <- xgb.DMatrix(x,label=y,weight=weights,missing=NA)
  # R XGBoost ignores params$seed; use its supported R RNG interface. Explicit
  # seeding here also makes fits invariant to which checkpoints were reused.
  set.seed(seed)
  params <- cfg$xgboost
  xgb.train(params=params,data=d,nrounds=cfg$nrounds,verbose=0)
}

gain <- function(model,names) {
  g <- setNames(rep(0,length(names)),names)
  imp <- xgb.importance(model=model)
  if (nrow(imp)) g[imp$Feature] <- imp$Gain
  g
}

prune_features <- function(x,gains,cfg,domains) {
  average <- colMeans(gains); frequency <- colMeans(gains>0)
  possible <- names(average)[frequency>=cfg$minimum_selection_frequency & average>cfg$minimum_mean_gain]
  possible <- possible[order(-average[possible],possible)]
  corr <- suppressWarnings(cor(x,use='pairwise.complete.obs'))
  retained <- character(); alternatives <- setNames(rep('',ncol(x)),colnames(x))
  for (f in possible) {
    redundant <- retained[vapply(retained,function(r) domains[[f]]==domains[[r]] &&
      is.finite(corr[f,r]) && abs(corr[f,r])>cfg$correlation_threshold,logical(1))]
    if (length(redundant)) alternatives[f]<-redundant[1] else retained<-c(retained,f)
  }
  table <- data.frame(feature=colnames(x),mean_gain=average,selection_frequency=frequency,
    retained=colnames(x)%in%retained,redundant_with=unname(alternatives),row.names=NULL)
  if (!length(retained)) stop('No stable features; no unpublished fallback applied')
  list(features=retained,table=table)
}

window_band <- function(curve,cfg) {
  g <- cfg$gam
  model <- gam(roc_auc~s(window,bs=g$basis,k=g$k),data=curve,method=g$method)
  days <- seq(cfg$window_min,cfg$window_max,by=g$grid_step)
  eps <- g$derivative_step
  value <- as.numeric(predict(model,newdata=data.frame(window=days)))
  derivative <- (as.numeric(predict(model,newdata=data.frame(window=days+eps)))-
                 as.numeric(predict(model,newdata=data.frame(window=days-eps))))/(2*eps)
  peak <- max(derivative); lo <- peak*g$low_fraction; hi <- peak*g$high_fraction
  eligible <- peak>0 & seq_along(days)>=which.max(derivative) & derivative>lo & derivative<hi
  bands <- sum(diff(c(FALSE,eligible))==1)
  lower <- if(any(eligible)) min(days[eligible]) else NA_real_
  upper <- if(any(eligible)) max(days[eligible]) else NA_real_
  selected <- if(any(eligible)) as.integer(floor((lower+upper)/2+0.5)) else NA_integer_
  list(curve=data.frame(window=days,smoothed_auc=value,slope=derivative,
       low_slope=lo,high_slope=hi,in_band=eligible),
       band=list(lower=lower,upper=upper,selected=selected,segments=bands,maximum_slope=peak,
         status=if(any(eligible)) 'inferred_with_disclosed_gam_settings' else 'no_qualifying_slope_band'))
}

bootstrap_auc <- function(y,p,cluster,n,seed) {
  set.seed(seed); code <- match(cluster,unique(cluster)); k <- max(code)
  values <- vapply(seq_len(n),function(i) {
    mult <- tabulate(sample.int(k,k,replace=TRUE),nbins=k)
    auc(y,p,mult[code])
  },numeric(1))
  list(low=unname(quantile(values,.025,na.rm=TRUE)),high=unname(quantile(values,.975,na.rm=TRUE)),
       valid=sum(is.finite(values)),values=values)
}

checkpoint <- function(path,signature,compute) {
  if (file.exists(path)) {
    old <- tryCatch(readRDS(path),error=function(e) NULL)
    if (!is.null(old) && identical(old$signature,signature)) return(old$value)
  }
  value <- compute(); temp<-paste0(path,'.tmp')
  saveRDS(list(signature=signature,value=value),temp); file.rename(temp,path)
  value
}

run <- function(input,config_file,directory) {
  cfg <- fromJSON(config_file,simplifyVector=TRUE); signature<-cfg$signature
  dir.create(directory,recursive=TRUE,showWarnings=FALSE)
  input_frame <- read.csv(input,stringsAsFactors=FALSE,check.names=FALSE)
  if (any(input_frame$season<2017 | input_frame$season>2025)) stop('Outside published study period')
  if (anyDuplicated(input_frame[c('game_id','pitcher_id')])) stop('Duplicate games')
  f <- input_frame
  f$case_season <- tolower(as.character(f$case_season))=='true'
  y_for <- function(frame,w) as.integer(frame$case_season & !is.na(frame$lead_days) & frame$lead_days>=1 & frame$lead_days<=w)
  train <- stratified_subset(y_for(f,cfg$window_max),1-cfg$test_fraction,cfg$seed)
  test <- setdiff(seq_len(nrow(f)),train)
  split <- f[c('game_id','pitcher_id','season','game_date')]; split$partition<-'test';split$partition[train]<-'train'
  write.csv(split,file.path(directory,'split.csv'),row.names=FALSE)
  tr<-f[train,]; te<-f[test,]; x<-as.matrix(tr[cfg$predictors]); storage.mode(x)<-'double'
  windows <- seq(cfg$window_min,cfg$window_max,by=cfg$window_step)
  all_cv <- list()
  for (w in windows) {
    values<-checkpoint(file.path(directory,paste0('window_',w,'.rds')),signature,function() {
      y<-y_for(tr,w); assignments<-folds(y,cfg$cv_folds,cfg$seed)
      do.call(rbind,lapply(seq_len(cfg$cv_folds),function(k) {
        fit<-which(assignments!=k); valid<-which(assignments==k)
        model<-fit_model(x[fit,,drop=FALSE],y[fit],cfg,cfg$seed+k)
        p<-predict(model,x[valid,,drop=FALSE])
        data.frame(window=w,fold=k,roc_auc=auc(y[valid],p),positive_games=sum(y[valid]),games=length(valid))
      }))
    })
    all_cv[[as.character(w)]]<-values
    if (w%%7==0) {cat('Window completed:',w,'\n');flush.console()}
  }
  cv<-do.call(rbind,all_cv);write.csv(cv,file.path(directory,'window_cv.csv'),row.names=FALSE)
  curve<-aggregate(roc_auc~window,cv,mean)
  band<-window_band(curve,cfg)
  write.csv(band$curve,file.path(directory,'gam_curve.csv'),row.names=FALSE)
  write_json(band$band,file.path(directory,'window_selection.json'),auto_unbox=TRUE,pretty=TRUE,na='null')
  if (is.na(band$band$selected)) {
    write_json(list(signature=signature,status='window_unavailable'),file.path(directory,'complete.json'),auto_unbox=TRUE)
    return(invisible(NULL))
  }
  w<-band$band$selected; y<-y_for(tr,w)
  gains<-list()
  for (r in seq_len(cfg$selection_repetitions)) {
    gains[[r]]<-checkpoint(file.path(directory,paste0('selection_',r,'.rds')),signature,function() {
      subset<-stratified_subset(y,cfg$selection_subset_fraction,cfg$seed+1000+r)
      assigns<-folds(y[subset],cfg$cv_folds,cfg$seed+r)
      do.call(rbind,lapply(seq_len(cfg$cv_folds),function(k) {
        ix<-subset[assigns!=k];model<-fit_model(x[ix,,drop=FALSE],y[ix],cfg,cfg$seed+r*10+k)
        gain(model,colnames(x))
      }))
    })
    if(r%%10==0) {cat('Stability repetition:',r,'of',cfg$selection_repetitions,'\n');flush.console()}
  }
  selected<-prune_features(x,do.call(rbind,gains),cfg,cfg$domains)
  write.csv(selected$table,file.path(directory,'feature_selection.csv'),row.names=FALSE)
  features<-selected$features
  model<-fit_model(x[,features,drop=FALSE],y,cfg,cfg$seed)
  xgb.save(model,file.path(directory,'model.bin'))
  write_json(list(signature=signature,selected_window=w,features=features,
    window_band=band$band,config=cfg),file.path(directory,'training_freeze.json'),auto_unbox=TRUE,pretty=TRUE,na='null')
  # Only now evaluate reserved games. No test-driven tuning, window or feature decisions.
  xtest<-as.matrix(te[features]); storage.mode(xtest)<-'double'
  yt<-y_for(te,w); p<-predict(model,xtest)
  ci<-bootstrap_auc(yt,p,paste(te$pitcher_id,te$season),cfg$bootstrap_iterations,cfg$seed)
  metrics<-data.frame(window=w,lower=band$band$lower,upper=band$band$upper,
    train_games=nrow(tr),test_games=nrow(te),positive_test_games=sum(yt),
    test_events=length(unique(na.omit(te$injury_event_id[yt==1]))),features=length(features),
    roc_auc=auc(yt,p),ci_low=ci$low,ci_high=ci$high,valid_bootstraps=ci$valid,
    shared_pitchers=length(intersect(unique(tr$pitcher_id),unique(te$pitcher_id))))
  write.csv(metrics,file.path(directory,'metrics.csv'),row.names=FALSE)
  predictions<-te[c('game_id','pitcher_id','season','game_date','injury_event_id')]
  predictions$target<-yt;predictions$prediction<-p
  write.csv(predictions,file.path(directory,'predictions.csv'),row.names=FALSE)
  # Published prior-region, previous-season sensitivity; no UCL-specific history.
  prior_features<-c(features,'prior_region_il')
  prior<-fit_model(as.matrix(tr[prior_features]),y,cfg,cfg$seed)
  pp<-predict(prior,as.matrix(te[prior_features]))
  pci<-bootstrap_auc(yt,pp,paste(te$pitcher_id,te$season),cfg$bootstrap_iterations,cfg$seed)
  write.csv(data.frame(roc_auc=auc(yt,pp),ci_low=pci$low,ci_high=pci$high,
      delta_auc=auc(yt,pp)-auc(yt,p)),file.path(directory,'prior_injury_sensitivity.csv'),row.names=FALSE)
  xgb.save(prior,file.path(directory,'prior_injury_model.bin'))
  # Mean total Gain across training pitcher-season bootstrap refits (Figure 4 style).
  cluster<-match(paste(tr$pitcher_id,tr$season),unique(paste(tr$pitcher_id,tr$season)))
  k<-max(cluster); boot_gain<-list()
  for (b in seq_len(cfg$importance_bootstrap_iterations)) {
    boot_gain[[b]]<-checkpoint(file.path(directory,paste0('importance_',b,'.rds')),signature,function() {
      set.seed(cfg$seed+10000+b);mult<-tabulate(sample.int(k,k,replace=TRUE),nbins=k)
      ix<-rep(seq_along(cluster),mult[cluster])
      if(length(unique(y[ix]))<2) return(setNames(rep(NA_real_,length(features)),features))
      fitted<-fit_model(x[ix,features,drop=FALSE],y[ix],cfg,cfg$seed+b)
      gain(fitted,features)
    })
    if(b%%100==0) {cat('Importance bootstrap:',b,'of',cfg$importance_bootstrap_iterations,'\n');flush.console()}
  }
  bg<-do.call(rbind,boot_gain)
  importance<-data.frame(feature=features,mean_gain=colMeans(bg,na.rm=TRUE),
                         gain_sd=apply(bg,2,sd,na.rm=TRUE),valid_refits=colSums(is.finite(bg)))
  importance<-importance[order(-importance$mean_gain),]
  write.csv(importance,file.path(directory,'importance.csv'),row.names=FALSE)
  writeLines(c(R.version.string,paste('xgboost',packageVersion('xgboost')),
               paste('mgcv',packageVersion('mgcv')),paste('jsonlite',packageVersion('jsonlite'))),file.path(directory,'software.txt'))
  write_json(list(signature=signature,status='complete',selection_fits=cfg$selection_repetitions*cfg$cv_folds,
    importance_refits=cfg$importance_bootstrap_iterations),file.path(directory,'complete.json'),auto_unbox=TRUE,pretty=TRUE)
  cat('Complete:',directory,'\n')
}

if (sys.nframe()==0) {
  args<-commandArgs(trailingOnly=TRUE)
  if(length(args)!=3) stop('Usage: Rscript scripts/kang_replication.R cohort.csv run_config.json output_directory')
  run(args[1],args[2],args[3])
}
