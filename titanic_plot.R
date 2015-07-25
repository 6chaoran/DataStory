##Exploratory Analysis
library(ggplot2)
library(grid)
library(gridExtra)
plot1=ggplot(train,aes(x=Survived,fill=Survived))+geom_histogram(aes(y=..count../sum(..count..)))+
theme_bw(base_size=18)+labs(y="Percent",title='Survival percentage',x="")
plot2=ggplot(train,aes(x=Sex,fill=Survived))+geom_histogram(aes(y=..count../sum(..count..)))+
theme_bw(base_size=18)+labs(y="Percent",title='Survival by Sex',x="")
plot3=ggplot(train,aes(x=Pclass,fill=Survived))+geom_histogram(aes(y=..count../sum(..count..)))+
theme_bw(base_size=18)+labs(y="Percent",title='Survival by Passenger Class',x="")
plot4=ggplot(train[train$Deck!='UNK',],aes(x=Deck,fill=Survived))+geom_histogram(aes(y=..count../sum(..count..)))+
theme_bw(base_size=18)+labs(y="Percent",title='Survival by Deck',x="")
png('data exploration1.png',width=800,height=600,units='px')
grid.arrange(plot1,plot2,plot3,plot4,ncol=2)
dev.off()

plot5=ggplot(train,aes(Age,fill=Survived))+geom_density(alpha=0.6)+
theme_bw(base_size = 18)+labs(title='Age Distribution')
plot6=ggplot(train,aes(Fare,fill=Survived))+geom_density(alpha=0.6)+
theme_bw(base_size = 18)+labs(title='Fare Distribution')
png('data exploration2.png',width=800,height=300,units='px')
grid.arrange(plot5,plot6,ncol=2)
dev.off()


##Decision Tree Plot
library(rpart)
library(rattle)
png('decisionTree1.png',width=800,height=600,units='px')
dTree1=rpart(factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FamilySize+Deck+CabinPos,data=train,
	control=rpart.control(cp=0.01))
fancyRpartPlot(dTree1,main='Decision Tree (cp=0.01)')
dev.off()

png('decisionTree2.png',width=800,height=600,units='px')
dTree2=rpart(factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FamilySize+Deck+CabinPos,data=train,
	control=rpart.control(cp=0.005))
fancyRpartPlot(dTree2,main='Decision Tree (cp=0.005)')
dev.off()


##Decision Tree Tuning
library(caret)
control=trainControl(method='repeatedcv',number=10,repeats=10)
rp.grid=expand.grid(cp=c(0.05,0.01,0.005,0.001))
dTree_tune=train(factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FamilySize+Deck+CabinPos,data=train,
	trControl=control,method='rpart',tuneGrid=rp.grid)
png('TreePruning.png',width=800,height=600,units='px')
plot(dTree_tune,main='Decision Tree Pruning')
dev.off()


##Random Forest Plot
library(randomForest)
rf1=randomForest(factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FamilySize+Deck+CabinPos,data=train,
	ntree=1000,mtry=10)
png('randomForest_errorrate.png',width=800,height=300,units='px')
par(mfrow=c(1,2))
plot(rf1,main='random Forest error rate')
varImpPlot(rf1,main='variable importance')
dev.off()


##Random Forest Tuning
control=trainControl(method='repeatedcv',number=10,repeats=10)
rf.grid=expand.grid(mtry=c(2,5,10,15,20))
rf_tune=train(factor(Survived)~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title+FamilySize+Deck+CabinPos,data=train,
	trControl=control,method='rf',tuneGrid=rf.grid)
png('Forest Tuning.png',width=800,height=600,units='px')
plot(rf_tune,main='Random Forest Tuning')
dev.off()

