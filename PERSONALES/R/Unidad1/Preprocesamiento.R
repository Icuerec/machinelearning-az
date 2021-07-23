df = read.csv("Unidad1/Data.csv")

#Tratamiento nan

df$Age = ifelse(is.na(df$Age),
                ave(df$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                df$Age)

df$Salary = ifelse(is.na(df$Salary),
                ave(df$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                df$Salary)

#Codificar variables categóricas

df$Country = factor(df$Country,
                    levels = c('France','Spain','Germany'),
                    labels = c(1,2,3))

df$Purchased = factor(df$Purchased,
                      levels = c('No','Yes'),
                      labels = c(0,1))

#Dividir los datos en entrenamiento y test

#install.packages('caTools')
library(caTools)
set.seed(123)

split = sample.split(df$Purchased,SplitRatio = 0.8)
training_set = subset(df,split==TRUE)
testing_set = subset(df,split == FALSE)

#Escalado de valores

training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[2:3])
