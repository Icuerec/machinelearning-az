#Regresión lineal múltiple

#Cargamos el archivo
df = read.csv('Unidad2_Regresion/RLM/50_Startups.csv')

#Codificar variables categóricas
df$State = factor(df$State,
                    levels = c('Florida','New York','California'),
                    labels = c(1,2,3))

#Dividir el conjunto en entrenamiento y test
library(caTools)

split = sample.split(df$Profit,SplitRatio = 0.8)
training_set = subset(df,split==TRUE)
testing_set = subset(df,split == FALSE)


#Ajustar el modelo de RLM con el conjunto de entrenamiento

regression = lm(formula = Profit ~ .,
                data = training_set)

#Haz summary en consola para ver como ha funcionado

#Predecir los resultados con el conjunto de testing

y_pred = predict(regression,testing_set)

#Optimizar nuestro conjunto de datos mediante la eliminación hacia atrás

install.packages("https://cran.r-project.org/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2015.6.26.2.tar.gz",repos=NULL, type="source")

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

