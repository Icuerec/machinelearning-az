#Regresión lineal simple

#Cargamos el archivo
dfSalary = read.csv('Unidad2_Regresion/RLS/Salary_Data.csv')

#Creamos el conjunto de test y el conjunto de entrenamiento
library(caTools)
set.seed(123)

split = sample.split(dfSalary$Salary,SplitRatio = 2/3)
training_set = subset(dfSalary,split==TRUE)
testing_set = subset(dfSalary,split == FALSE)

#Ajustar el modelo de regresión lineal simple con el training_set

regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Predecir resultados con el set de test (Se tienen que llamar igual las columnas!!!)

y_pred = predict(regressor, newdata = testing_set)

#Visualizar datos y predicción

#install.packages('ggplot2')

library(ggplot2)

ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'orange') +
  geom_line(aes(x = training_set$YearsExperience,
                y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Sueldo vs Años exp. (Conjunto de Entrenamiento)') +
  xlab('Años exp.') + 
  ylab('Sueldo')

