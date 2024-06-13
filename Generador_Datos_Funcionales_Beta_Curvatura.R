set.seed(100)
Simulaciones = 200 #Número de simulaciones
N <- 125 # Tamaño de las muestras
m <- 50  # Número de nodos

####################### LOGARITMO ######################################

T<- seq(0.01, 1, length = m)
T2 <- seq(0.01, 1, length = 500)
betaT<- (log(T) + 5)/5 - T^2
betaT2<- (log(T2) + 5)/5 - T2^2
plot(T,betaT, type='l') 

#############################################################################


####################### DENSIDAD PARETO #####################################
pareto_pdf <- function(x, b = 1, scale = 1, loc = 1){                       #
  density <- b / ((x-loc)/scale)^(b + 1)                                    #
  return(density)}                                                          #
T<- seq(0.05, 1, length = m)
T2 <- seq(0.05, 1, length = 500)
betaT<-(pareto_pdf(T, b =1, loc=-0.55, scale=0.7)+T^2 - 0.7)/3               #
betaT2<-(pareto_pdf(T2, b =1, loc=-0.55, scale=0.7)+T2^2 - 0.7)/3 
plot(T,betaT, type='l')                                                     #
#############################################################################

rankX <- 20  # Esto es k
Phi <- cbind(1 / sqrt(m), poly(T, degree = rankX - 1))  # Base ortogonal de polinomios k-1
lambda <- rankX:1

# Crear matrices para almacenar las muestras Monte Carlo
X_se_Muestras <- array(0, dim = c(N, Simulaciones, m))  # Dimensiones: (N, número de muestras, tamaño de la muestra)
y_Muestras <- matrix(data = 0, nrow = Simulaciones, ncol = N)

for (i in 1:Simulaciones) {
  eps <- rnorm(N, 0, 0.01)  # Residuos generados N(0,0.05)
  Xi <- sapply(lambda, function(l) scale(rnorm(N, sd = sqrt(l)), scale = FALSE))
  X <- Xi %*% t(Phi) 
  lfterm <- as.numeric((1 / m) * X %*% betaT)
  alpha <- 0.15  # Alpha puesto de base
  y <- alpha + lfterm + eps  # Creamos la variable respuesta
  y_Muestras[i,] = y
  X_se_Muestras[, i, ] <- matrix(X, nrow = N, ncol = m)  # Almacenamos la muestra Monte Carlo en la matriz resultados
}

######################## GUARDAR LOS DATOS GENERADOS #########################################

saveRDS(object = X_se_Muestras, file = "X_se_Muestras_Pareto_Curvatura_001_125.rds")
Importacion <- readRDS(file = "X_se_Muestras_Pareto_Curvatura_001_125.rds")
write.csv(betaT, file = "Beta_Muestras_Pareto_Curvatura_001_125.csv", row.names = FALSE)
write.csv(T, file = "T_Muestras_Pareto_Curvatura_001_125.csv", row.names = FALSE)
write.csv(y_Muestras, file = "y_Muestras_Pareto_Curvatura_001_125.csv", row.names = FALSE)

write.csv(T2, file = "T2_Muestras_Pareto_Curvatura.csv", row.names = FALSE)
write.csv(betaT2, file = "Beta2_Muestras_Pareto_Curvatura.csv", row.names = FALSE)

#############################################################################################


ex3 = function(){
  A = matrix(c(1,2,3,4) , nrow = 2, ncol = 2)
  print(A)
}

ex3()
