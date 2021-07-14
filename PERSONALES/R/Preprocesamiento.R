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
