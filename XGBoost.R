# script to determine the sleep state of an animal based on epochs from data 
# obtained from Pinnacle EEG devices. This script leverages XGBoost as a classifier
# and accepts inputs as power_scores.tsv files from pinnacle

library(tidyverse)
library(xgboost)
library(caret)
library(mixtools)
library(dplyr)

pre_process=function(input_file)
{
  #read file
  df1=read.delim(input_file,sep = '\t')
  #extract relevant rows and rename
  name_index=grep('Date',df1[,1])
  names(df1)=c(df1[name_index,1:ncol(df1)-1],'class')
  df1$class=0;
  params=as.data.frame(cbind(state=df1$Date[5:8],class=df1$Time[5:8]));
  df1=df1[(name_index+1):nrow(df1),]
  df1=as.data.frame(lapply(df1,as.numeric))
  
  #split into electrode 1 and 2, perform feature calculation
  eeg1=df1[,grep('EEG1',names(df1))];eeg2=df1[,grep('EEG2',names(df1))]
  perms_eeg1=combn(ncol(eeg1),2)
  perms_eeg2=combn(ncol(eeg2),2)
  eeg1_names=c(names(eeg1),paste(names(eeg1)[perms_eeg1[1,]],names(eeg1)[perms_eeg1[2,]],sep='_')) 
  eeg2_names=c(names(eeg2),paste(names(eeg2)[perms_eeg2[1,]],names(eeg2)[perms_eeg2[2,]],sep='_')) 
  eeg1=eeg1 %>% {cbind(.,(eeg1[,perms_eeg1[1,]]/eeg1[,perms_eeg1[2,]]))}%>% setNames(eeg1_names)
  eeg2=eeg2 %>% {cbind(.,(eeg2[,perms_eeg2[1,]]/eeg2[,perms_eeg2[2,]]))}%>% setNames(eeg2_names)
  
  #calculate relative power as features
  eeg1=eeg1 %>% mutate(across(c('EEG1_Alpha', 'EEG1_Beta','EEG1_Gamma','EEG1_Delta','EEG1_Theta'),~./EEG1_Full,.names="{col}_relative"))
  eeg2=eeg2 %>% mutate(across(c('EEG2_Alpha', 'EEG2_Beta','EEG2_Gamma','EEG2_Delta','EEG2_Theta'),~./EEG2_Full,.names="{col}_relative"))

  #find column where scoring is contained
  score_index=grep('Numeric',names(df1))
  df_out=df1 %>% select(contains('EMG')) %>% cbind(eeg1,eeg2,.,class=df1[,score_index])
  df_out$class=gsub(255,0,df_out$class);
  df_out$class=as.numeric(df_out$class);
  #remove NAs
  df_out[!df_out$EEG1_Alpha_EEG1_Beta_Ratio=='NaN',];
  #add next and previous row to each feature
  #df_out=cbind(df_out,lag(df_out),lead(df_out)) %>% setNames(c(names(df_out),paste(names(df_out),'_prev',sep=''),paste(names(df_out),'_next',sep='')))
  #df_out=df_out[c(-1,-nrow(df_out)),]
  return(df_out);
}

estimate_error=function(input_file,model,softprob)
{
  data_in=pre_process(input_file);
  labels=data_in$class;
  data_in=as.matrix(data_in[,model$feature_names]);
  predictions=predict(model,data_in);
  if (softprob){
  num_class=model$params$num_class;
  dim(predictions)=c(num_class,(length(predictions)/num_class))
  predictions=t(predictions) %>% as.data.frame() %>% setNames(c('0','1','2','3'))
  scores=as.numeric(names(predictions)[max.col(predictions,'first')])
  return(sum(labels!=scores)/nrow(predictions));}
  else{sum(labels!=predictions)/nrow(predictions)}
}

predict_file=function(input_file,model,features,softprob)
{
  #read file and pre-process
  eeg_in=pre_process(input_file);
  test2=as.matrix(eeg_in[,features]);
  pred=predict(model,as.matrix(test2),reshape=T)
  
  if (softprob!=1){
  #write to tsv file
  pred=gsub(0,255,pred);
  raw_file=read.delim(input_file,sep = '\t')
  score_index=grep('Numeric',raw_file[10,])
  raw_file[10,score_index]='ML_scored_Numeric';
  raw_file[11:(10+length(pred)),score_index]=pred;}
  else{
    low_index=which((apply(pred,1,max)<0.5))+10;
    mid_index=which((apply(pred,1,max)<0.75 & apply(pred,1,max)>0.5))+10;
    pred=(apply(pred,1,which.max))-1
    pred=gsub(0,255,pred);
    
    raw_file=read.delim(input_file,sep = '\t')
    score_index=grep('Numeric',raw_file[10,])
    raw_file[10,score_index]='ML_scored_Numeric';
    raw_file[11:(10+length(pred)),score_index]=pred;
    indices=cbind(low_index=c(low_index,rep('',length(mid_index)-length(low_index))),mid_index=mid_index)
    indices=indices=1;
    }
  rbind(names(raw_file),raw_file);
  raw_file=apply(raw_file,c(1,2),na.omit);
  write.table(raw_file,file=gsub('power_scores.tsv','scored.tsv',input_file),sep='\t',row.names = F,col.names = F)
  write.table(indices,file=gsub('power_scores.tsv','epochs_to_check.csv',input_file),row.names = F,sep=',')
  }

create_dense_mat=function (data)
{
  #convert to numeric and remove last
  xg_data=as.data.frame(apply(data,c(1,2),as.numeric))
  #xg_data=xg_data[-1,]
  
  #split 30-70%, label as test and train
  train_index=createDataPartition(xg_data$class,p=0.7)
  xg_train=xg_data[train_index$Resample1,];
  xg_train_label=xg_train$class;
  xg_train=xg_train %>% select(!contains('class'))
  xg_train_Dmat=xgb.DMatrix(data=as.matrix(xg_train),label=xg_train_label)
  
  xg_test=xg_data[-train_index$Resample1,];
  xg_test_label=xg_test$class;
  xg_test=xg_test %>% select(!contains('class'))
  return(list(xg_train_Dmat,xg_train,xg_train_label,xg_test,xg_test_label))
}

#Read multiple files and combine 
wd='example/directory'
setwd(wd)
dataset='';
files=list.files('.','power')
dataset=bind_rows(lapply(files,pre_process))

#Create variables for training and testing 
xg_Dmat=create_dense_mat(dataset)
xg_train_Dmat=xg_Dmat[[1]];

#Calculate class weights for rebalancing
weights=length(xg_Dmat[[3]])/table(xg_Dmat[[3]]);
xg_train=xg_Dmat[[2]]
xg_train_label=xg_Dmat[[3]]
xg_test=xg_Dmat[[4]]
xg_test_label=xg_Dmat[[5]]

#set model hyperparameters
params=list(booster='gbtree',objective='multi:softprob',eval_metric='merror',eta=0.1,gamma=4.5,max_depth=8,num_class=length(unique(xg_Dmat[[3]])),subsample=0.7)

# grid search
# grid_params=expand.grid(nrounds=seq(1000),max_depth=c(3:15),eta=seq(0.1,0.5,0.1),gamma=seq(0,5,0.5),subsample=seq(0,1,0.2),min_child_weight=1,colsample_bytree=1)
# train_control = trainControl(method = "cv", number = 10, search = "grid",verboseIter = T)
# xgcv=caret::train(x=as.matrix(xg_test),y=xg_test_label,method='xgbTree',verbosity=0,metric='mlogloss',tuneGrid=grid_params,trControl=train_control,verbose=F)

#train model 
xgcv=xgb.cv(data=xg_train_Dmat,params=params,nrounds=500,prediction = T,early_stopping_rounds = 20,nfold=5)
nrounds=xgcv$best_iteration

set.seed(123);
watchlist=list(train=xg_train_Dmat,test=xgb.DMatrix(data=as.matrix(xg_test),label=xg_test_label))
#train with class weights
xgb_mod=xgb.train(params=params,data=xg_train_Dmat,nrounds=nrounds,scale_pos_weight=weights,early_stopping_rounds = 20,watchlist=watchlist)

#train without class weights
xgb_mod=xgb.train(params=params,data=xg_train_Dmat,nrounds=nrounds,early_stopping_rounds = 20,watchlist=watchlist)

xg_scores=predict(xgb_mod,as.matrix(xg_test))
err_rate=sum(xg_scores!=xg_test_label)/length(xg_scores)
print(err_rate)
feature_importance=xgb.importance(colnames(xg_Dmat[[3]]),xgb_mod)

#Reduce features
features_to_use=feature_importance$Feature[feature_importance$Gain>0.01]
xg_train2=xg_train[,names(xg_train) %in% features_to_use]
xg_train_Dmat2=xgb.DMatrix(data=as.matrix(xg_train2),label=xg_train_label)
xg_test2=xg_test[,names(xg_train) %in% features_to_use]
watchlist=list(train=xg_train_Dmat2,test=xgb.DMatrix(data=as.matrix(xg_test2),label=xg_test_label))
#train without class weights
xgb_mod2=xgb.train(params=params,data=xg_train_Dmat2,nrounds=nrounds,early_stopping_rounds = 20,watchlist=watchlist)
#Train with class weights
xgb_mod2=xgb.train(params=params,data=xg_train_Dmat2,nrounds=nrounds,scale_pos_weight=weights,early_stopping_rounds = 20,watchlist=watchlist)

#save models
setwd('example/dir')
filename='example'
xgb.save(xgb_mod,paste(filename,'.model',sep=''))
write.table(xgb_mod$feature_names,paste(filename,'.features',sep=''))
features=xgb_mod$feature_names;

#load models
xgb_mod2=xgb.load('example.model')
features=read.table('example.features')$x

#estimate error based on test files
setwd('example/directory')
file_list=dir()[grep('power',dir())]
mean_error=lapply(file_list,estimate_error,xgb_mod2,1) %>% {unlist(.,recursive=F)}

input='example_input_file.tsv'
estimate_error(input,xgb_mod2,1)

setwd('example_directory')
predict_file('input_file.tsv',xgb_mod,features,1)