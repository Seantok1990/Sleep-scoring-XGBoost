#random forest classifier for EEG to determine sleep states using a Pinnacle 
#wireless EEG system using power scores processed using Sirenia software.

library(randomForest)
library(eegkit)
library(caret)
library(dplyr)

split_data=function(input_data,label,p)
{
  eval(parse(text=paste('input.index=createDataPartition(y=input_data$',label,',p=',p,',list=F)',sep='')))
  train=input_data[input.index,]
  test=setdiff(input_data,input_data[input.index,])
  return(c(list(train),list(test)))
}
pre_process=function(input_file)
{
  #read file in the format of 
  df1=read.delim(input_file,sep = '\t')
  #extract relevant rows
  names(df1)=c(df1[10,1:ncol(df1)-1],'state')
  df1$class=0;
  df1=df1[11:nrow(df1),]
  df1=as.data.frame(lapply(df1,as.numeric))
  #split into electrode 1 and 2, perform feature engineering
  eeg1=df1[grep('EEG1',names(df1))] %>% .[,2:ncol(.)];eeg2=df1[grep('EEG2',names(df1))] %>% .[,2:ncol(.)];
  perms=combn(ncol(eeg1),2);colnames=paste(names(eeg1)[perms[1,]],names(eeg1)[perms[2,]],sep='_')
  colnames_eeg1=c(names(eeg1),paste(names(eeg1)[perms[1,]],names(eeg1)[perms[2,]],'Ratio',sep='_'))
  colnames_eeg2=gsub('EEG1','EEG2',colnames_eeg1);
  eeg1=cbind(eeg1,eeg1[,perms[1,]]/eeg1[,perms[2,]]);eeg2=cbind(eeg2,eeg2[,perms[1,]]/eeg2[,perms[2,]])
  names(eeg1)=colnames_eeg1;names(eeg2)=colnames_eeg2;
  #return combined feature set
  
  df_out=df1 %>% select(contains('EMG')) %>% cbind(eeg1,eeg2,.,state=df1 %>% select(contains('Numeric')) %>% .[,1])
  df_out$state=gsub(255,0,df_out$state)
  df_out$state =as.character(df_out$state) %>% as.factor
  return(df_out);
}
setwd('example\directory')
input_file='example_power_scores.tsv'

#preprocess and generate features
dataset=pre_process(input_file)
dataset=na.omit(dataset)

#generate a random forest model based on a training dataset with grid search
# dataset=(dataset[,1:(ncol(dataset)-1)])
control <- trainControl(method="repeatedcv", number=10, repeats=3)
metric='accuracy'
mtry=1:10
ntree=seq(1,1001,50)
maxnodes=1:10
tunegrid=expand.grid(.mtry=c(1:20))
rf_default <- train(class~., data=dataset, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
