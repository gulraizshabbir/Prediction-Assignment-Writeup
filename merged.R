testset<-read.table("X_test.txt")
testlabels<-read.table("y_test.txt")
    colnames(testlabels)<-c("names")
    columns<-read.table("features.txt")
    columns<-as.character(columns[,2])
    colnames(testset)<-columns
    testset<-cbind(testset,testlabels)
activity<-read.table("activity_labels.txt")
    colnames(activity)<-c("names","Activity")
library("plyr", lib.loc="~/R/win-library/3.1")
library("dplyr")
    merged<-arrange(join(testset,activity),names)
##grouped<-group_by(merged,Activity)

    splited<-split(merged,merged$Activity)
        means<-lapply(splited,function(x)lapply(x[,1:562],mean))
        sd<-lapply(splited,function(x)lapply(x[,1:562],sd))
        sd1<-unlist(sd)
        means1<-unlist(means)

    final<-rbind(means1,sd1)## data set showing mean and standard deviation in rows, for every variable, seperately for every activity.
    lists<-rbind(means,sd)## gives list of 561 variables with standard deviation n mean seperately.

    newmeans<-sapply(splited,function(x)lapply(x[,1:562],mean))
    newsd<-sapply(splited,function(x)lapply(x[,1:562],sd))
    colnames(newmeans)<-c("LAYING(mean)","SITTING(mean)","STANDING(mean)","WALKING(mean)"," WALKING_DOWNSTAIRS(mean)","WALKING_UPSTAIRS(mean)")
    colnames(newsd)<-c("LAYING(sd)","SITTING(sd)","STANDING(sd)","WALKING(sd)"," WALKING_DOWNSTAIRS(sd)","WALKING_UPSTAIRS(sd)")

    finaldataset<-cbind(newmeans,newsd)
    head(finaldataset)

##use show to check the list for x=mean,average,sd   y=ACTIVITY.....it will return the corresponding table
show<-function(x,y){
    if(x==mean){i=1}
    if(x==average){i=1}
    if(x==sd){i=2}
    if(y=="LAYING"){j=1}
    if(y=="SITTING"){j=2}
    if(y=="STANDING"){j=3}
    if(y=="WALKING"){j=4}
    if(y=="WALKING_DOWNSTAIRS"){j=5}
    if(y=="WALKING_UPSTAIRS"){j=6}
    lists[[i,j]]
}