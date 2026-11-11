# Change the outer partition only; import the frozen fitting/statistics functions.
source('scripts/kang_replication.R')

target_for <- function(f,w) as.integer(tolower(as.character(f$case_season))=='true' &
  !is.na(f$lead_days) & f$lead_days>=1 & f$lead_days<=w)

train_chronological <- function(input,config_file,directory) {
 cfg<-fromJSON(config_file); signature<-cfg$signature
 tr<-read.csv(input,stringsAsFactors=FALSE,check.names=FALSE)
 stopifnot(all(as.Date(tr$game_date)>=as.Date('2017-01-01')),
           all(as.Date(tr$game_date)<=as.Date('2023-12-31')),
           !anyDuplicated(tr[c('game_id','pitcher_id')]))
 x<-as.matrix(tr[cfg$predictors]);storage.mode(x)<-'double'
 windows<-seq(cfg$window_min,cfg$window_max,by=cfg$window_step)
 support<-do.call(rbind,lapply(windows,function(w) {
  y<-target_for(tr,w)
  data.frame(window=w,games=nrow(tr),positive_games=sum(y),negative_games=sum(1-y),
    represented_events=length(unique(na.omit(tr$injury_event_id[y==1]))))
 }))
 write.csv(support,file.path(directory,'training_support.csv'),row.names=FALSE)
 if(any(support$positive_games<cfg$cv_folds | support$negative_games<cfg$cv_folds)) {
  write_json(list(signature=signature,status='insufficient_training_support'),file.path(directory,'complete.json'),auto_unbox=TRUE)
  return(invisible(NULL))
 }
 all_cv<-list()
 for(w in windows) {
  values<-checkpoint(file.path(directory,paste0('window_',w,'.rds')),signature,function() {
   y<-target_for(tr,w);assignments<-folds(y,cfg$cv_folds,cfg$seed)
   do.call(rbind,lapply(seq_len(cfg$cv_folds),function(k) {
    fit<-which(assignments!=k);valid<-which(assignments==k)
    model<-fit_model(x[fit,,drop=FALSE],y[fit],cfg,cfg$seed+k)
    p<-predict(model,x[valid,,drop=FALSE])
    data.frame(window=w,fold=k,roc_auc=auc(y[valid],p),positive_games=sum(y[valid]),games=length(valid))
   }))
  })
  all_cv[[as.character(w)]]<-values
  if(w%%7==0) {cat('Training window:',w,'\n');flush.console()}
 }
 cv<-do.call(rbind,all_cv);write.csv(cv,file.path(directory,'window_cv.csv'),row.names=FALSE)
 curve<-aggregate(roc_auc~window,cv,mean);band<-window_band(curve,cfg)
 write.csv(band$curve,file.path(directory,'gam_curve.csv'),row.names=FALSE)
 write_json(band$band,file.path(directory,'window_selection.json'),auto_unbox=TRUE,pretty=TRUE,na='null')
 if(is.na(band$band$selected)) {
  write_json(list(signature=signature,status='window_unavailable'),file.path(directory,'complete.json'),auto_unbox=TRUE)
  return(invisible(NULL))
 }
 w<-band$band$selected;y<-target_for(tr,w);gains<-list()
 for(r in seq_len(cfg$selection_repetitions)) {
  gains[[r]]<-checkpoint(file.path(directory,paste0('selection_',r,'.rds')),signature,function() {
   subset<-stratified_subset(y,cfg$selection_subset_fraction,cfg$seed+1000+r)
   assigns<-folds(y[subset],cfg$cv_folds,cfg$seed+r)
   do.call(rbind,lapply(seq_len(cfg$cv_folds),function(k) {
    ix<-subset[assigns!=k];model<-fit_model(x[ix,,drop=FALSE],y[ix],cfg,cfg$seed+r*10+k)
    gain(model,colnames(x))
   }))
  })
  if(r%%10==0) {cat('Selection repetition:',r,'\n');flush.console()}
 }
 selected<-prune_features(x,do.call(rbind,gains),cfg,cfg$domains)
 write.csv(selected$table,file.path(directory,'feature_selection.csv'),row.names=FALSE)
 features<-selected$features;model<-fit_model(x[,features,drop=FALSE],y,cfg,cfg$seed)
 xgb.save(model,file.path(directory,'model.bin'))
 write_json(list(signature=signature,selected_window=w,features=features,window_band=band$band,
   training_min_date=min(tr$game_date),training_max_date=max(tr$game_date),
   positive_games=sum(y),negative_games=sum(1-y),
   positive_weight=length(y)/(2*sum(y)),negative_weight=length(y)/(2*sum(1-y)),config=cfg),
   file.path(directory,'training_freeze.json'),auto_unbox=TRUE,pretty=TRUE,na='null')
 cluster<-match(paste(tr$pitcher_id,tr$season),unique(paste(tr$pitcher_id,tr$season)))
 k<-max(cluster);boot_gain<-list()
 for(b in seq_len(cfg$importance_bootstrap_iterations)) {
  boot_gain[[b]]<-checkpoint(file.path(directory,paste0('importance_',b,'.rds')),signature,function() {
   set.seed(cfg$seed+10000+b);mult<-tabulate(sample.int(k,k,replace=TRUE),nbins=k)
   ix<-rep(seq_along(cluster),mult[cluster])
   if(length(unique(y[ix]))<2)return(setNames(rep(NA_real_,length(features)),features))
   fitted<-fit_model(x[ix,features,drop=FALSE],y[ix],cfg,cfg$seed+b)
   gain(fitted,features)
  })
  if(b%%100==0) {cat('Training importance bootstrap:',b,'\n');flush.console()}
 }
 bg<-do.call(rbind,boot_gain)
 importance<-data.frame(feature=features,mean_gain=colMeans(bg,na.rm=TRUE),gain_sd=apply(bg,2,sd,na.rm=TRUE),
                        valid_refits=colSums(is.finite(bg)))
 importance<-importance[order(-importance$mean_gain),]
 write.csv(importance,file.path(directory,'importance.csv'),row.names=FALSE)
 write_json(list(signature=signature,status='complete',selection_fits=cfg$selection_repetitions*cfg$cv_folds,
   importance_refits=cfg$importance_bootstrap_iterations),file.path(directory,'complete.json'),auto_unbox=TRUE,pretty=TRUE)
 cat('Chronological training complete\n')
}

evaluate_chronological <- function(input,config_file,directory,period,cutoff) {
 cfg<-fromJSON(config_file);freeze<-fromJSON(file.path(directory,'training_freeze.json'))
 f<-read.csv(input,stringsAsFactors=FALSE,check.names=FALSE)
 bounds<-list(training=c('2017-01-01','2023-12-31'),validation=c('2024-01-01','2025-12-31'),test=c('2026-01-01','2026-12-31'))
 stopifnot(period%in%names(bounds),all(as.Date(f$game_date)>=as.Date(bounds[[period]][1])),
  all(as.Date(f$game_date)<=as.Date(bounds[[period]][2])))
 model<-xgb.load(file.path(directory,'model.bin'))
 x<-as.matrix(f[freeze$features]);storage.mode(x)<-'double';p<-predict(model,x)
 y<-target_for(f,freeze$selected_window)
 known_positive<-y==1 & !is.na(f$lead_days) & as.Date(f$game_date)+f$lead_days<=as.Date(cutoff)
 observed<-known_positive | as.Date(f$game_date)+freeze$selected_window<=as.Date(cutoff)
 y[!observed]<-NA_integer_
 pred<-f[c('game_id','pitcher_id','season','game_date','injury_event_id')]
 pred$target<-y;pred$prediction<-p
 write.csv(pred,file.path(directory,paste0('predictions_',period,'.csv')),row.names=FALSE)
 keep<-!is.na(y);yy<-y[keep];pp<-p[keep]
 ci<-list(low=NA_real_,high=NA_real_,valid=0)
 if(length(unique(yy))==2)ci<-bootstrap_auc(yy,pp,paste(f$pitcher_id[keep],f$season[keep]),cfg$bootstrap_iterations,cfg$seed)
 m<-data.frame(period=period,window=freeze$selected_window,scored_games=nrow(f),eligible_games=sum(keep),
  censored_games=sum(!keep),positive_games=sum(yy),negative_games=sum(1-yy),prevalence=mean(yy),
  represented_events=length(unique(na.omit(f$injury_event_id[which(y==1)]))),
  roc_auc=auc(yy,pp),ci_low=ci$low,ci_high=ci$high,valid_bootstraps=ci$valid,
  brier=mean((pp-yy)^2),mean_risk_bias=mean(pp)-mean(yy))
 write.csv(m,file.path(directory,paste0('metrics_',period,'.csv')),row.names=FALSE)
 cat('Scored frozen model:',period,'\n')
}

if(sys.nframe()==0) {
 args<-commandArgs(trailingOnly=TRUE)
 if(args[1]=='train' && length(args)==4) train_chronological(args[2],args[3],args[4])
 else if(args[1]=='score' && length(args)==6) evaluate_chronological(args[2],args[3],args[4],args[5],args[6])
 else stop('Use train input config directory; or score input config directory period cutoff')
}
