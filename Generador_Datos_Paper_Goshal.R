set.seed(1)
Simulaciones = 200 #Número de simulaciones
N <- 100 # Tamaño de las muestras
m <- 50  # Número de nodos

####################### DATOS PAPER ######################################

T <- seq(0, 1, length = m) # Espacio en el que vamos a trabajar
betaT<- 0.1 * sin(pi * T) # Simulación Normal

T2 <- seq(0,1 , length = 500)
betaT2 <- 0.1 * sin(pi * T2)
rankX <- 20  # Esto es k
Phi <- cbind(1 / sqrt(m), poly(T, degree = rankX - 1))  # Base ortogonal de polinomios k-1
lambda <- rankX:1

# Crear matrices para almacenar las muestras Monte Carlo
X_se_Muestras <- array(0, dim = c(N, Simulaciones, m))  # Dimensiones: (N, número de muestras, tamaño de la muestra)
y_Muestras <- matrix(data = 0, nrow = Simulaciones, ncol = N)

for (i in 1:Simulaciones) {
  eps <- rnorm(N, 0, 0.05)  # Residuos generados N(0,0.05)
  Xi <- sapply(lambda, function(l) scale(rnorm(N, sd = sqrt(l)), scale = FALSE))
  X <- Xi %*% t(Phi) 
  lfterm <- as.numeric((1 / m) * X %*% betaT)
  alpha <- 0.15  # Alpha puesto de base
  y <- alpha + lfterm + eps  # Creamos la variable respuesta
  y_Muestras[i,] = y
  X_se_Muestras[, i, ] <- matrix(X, nrow = N, ncol = m)  # Almacenamos la muestra Monte Carlo en la matriz resultados
}

######################## GUARDAR LOS DATOS GENERADOS #########################################

saveRDS(object = X_se_Muestras, file = "X_se_Muestras_Goshal_005_100.rds")
Importacion <- readRDS(file = "X_se_Muestras_Goshal_005_100.rds")
write.csv(betaT, file = "Beta_Muestras_Goshal.csv", row.names = FALSE)
write.csv(betaT2, file = "Beta2_Muestras_Goshal.csv", row.names = FALSE)
write.csv(T, file = "T_Muestras_Goshal_.csv", row.names = FALSE)
write.csv(y_Muestras, file = "y_Muestras_Goshal_005_100.csv", row.names = FALSE)
write.csv(T2, file = "T2_Muestras_Goshal.csv", row.names = FALSE)

#############################################################################################