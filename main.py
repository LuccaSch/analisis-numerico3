import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import cg  # Gradiente Conjugado
import sympy as sp
#tiempo de ejecucion para comparar en el inciso c y d

T1=[]
T2=[]
it1=[]
it2=[]


# Funciones de ejemplo y sus derivadas
def f(x):
    return x**3 - x - 2

def f_prime(x):
    return 3*x**2 - 1

def f_double_prime(x):
    return 6*x

#-------------------------------------------------A-------------------------------------------------

# Método de Newton-Secante
def newton_secant_method(a, b, tol=1e-8, max_iter=50):
  
    for i in range(max_iter):
        f_a = f(a)
        f_b = f(b)
        
        if abs(f_b)<tol:
            return b, i+1
        
        f_prime_mid=f_prime((a+b)/2)
        peso=f_double_prime(a)/(f_double_prime(a) + f_double_prime(b)) 
        
        pendiente_secante=(f_b-f_a)/(b-a)
        M_NS = peso * f_prime_mid + (1 - peso) * pendiente_secante
        nuevo=b-f_b / M_NS
        if abs(nuevo-b) < tol:
            return nuevo, i+1
        
        a, b = b, nuevo
    
    raise RuntimeError("El método de Newton-Secante no converge")

# Intervalo inicial
a, b = 1, 2
raiz_ns, iteraciones_ns = newton_secant_method(a, b)

# Gráfico
x_vals = np.linspace(a, b, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='$f(x) = x^3 - x - 2$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(raiz_ns, color='red', linestyle='--', label=f'Raíz aproximada: {raiz_ns:.6f}')
plt.title('Método Newton-Secante')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

#-------------------------------------------------B-------------------------------------------------
# Funciones adicionales
def newton_raphson_method(x0, func, tol=1e-16, max_iter=50):
    x = x0
    for i in range(max_iter):
        f_valor=func(x)
        f_derivada_valor = f_prime(x) 
        if f_derivada_valor==0:
            print("Metodo de Newton-Raphson falla.")
            return np.nan
        x_new = x - f_valor / f_derivada_valor
        if abs(x_new - x) < tol:
            return x_new, i
        x = x_new
   

def regula_falsi_method(a, b, tol=1e-9, max_iter=50):
    for i in range(max_iter):
        fa = f(a)
        fb = f(b)
        c = (b*fb - a*fa) / (fb - fa)
        fc = f(c)
        if abs(fc) < tol:
            return c, i
        if fa * fc < 0:
            b = c
        else:
            a = c
    return c, max_iter

# Comparación de métodos
x0 = (a + b) / 2
root_nr, iteraciones_nr = newton_raphson_method(x0, f)  # Se pasa 'f' como argumento
root_rf, iteraciones_rf = regula_falsi_method(a, b)

print(f"Raíz con Newton-Raphson: {root_nr:.6f}, Iteraciones: {iteraciones_nr}")
print(f"Raíz con Regula-Falsi: {root_rf:.6f}, Iteraciones: {iteraciones_rf}")
print(f"Raíz con Newton-Secante: {raiz_ns:.6f}, Iteraciones: {iteraciones_ns}")

#-------------------------------------------------C-------------------------------------------------


# Función de electroneutralidad
def electroneutralidad(H, C1, pKa1, C2, pKa2):
    return H - 10**-14 / H + C2 * H / (10**-pKa2 + H) - C1 / (10**pKa1 * H + 1)

# Derivada de la función de electroneutralidad
def derivada_electroneutralidad(H, C1, pKa1, C2, pKa2):
    term1 = 1
    term2 = 10**-14 / H**2
    term3 = C2 * 10**-pKa2 / (10**-pKa2 + H)**2
    term4 = C1 * 10**pKa1 / (10**pKa1 * H + 1)**2
    return term1 + term2 + term3 + term4

# Segunda derivada de la función de electroneutralidad
def segunda_derivada_electroneutralidad(H, C1, pKa1, C2, pKa2):
    term1 = 0
    term2 = (-2 * 10**-14) / H**3
    term3 = -2 * C2 * 10**-pKa2 / (10**-pKa2 + H)**3
    term4 = -2 * C1 * 10**pKa1 / (10**pKa1 * H + 1)**3
    return term1 + term2 + term3 + term4

# Método de Newton-Secante
def newton_secant_method_electroneutralidad(H_a, H_b, Ca, pKa1, Cb, pKa2, tol):
    max_iter = 50
    for i in range(max_iter):
        f_a = electroneutralidad(H_a, Ca, pKa1, Cb, pKa2)
        f_b = electroneutralidad(H_b, Ca, pKa1, Cb, pKa2)
        
        if abs(f_b) < tol:
            return H_b, i+1
        
        # Calculando las pendientes
        f_prime_mid = derivada_electroneutralidad((H_a + H_b) / 2, Ca, pKa1, Cb, pKa2)
        peso = segunda_derivada_electroneutralidad(H_a, Ca, pKa1, Cb, pKa2) / (segunda_derivada_electroneutralidad(H_a, Ca, pKa1, Cb, pKa2) + segunda_derivada_electroneutralidad(H_b, Ca, pKa1, Cb, pKa2))
        
        pendiente_secante = (f_b - f_a) / (H_b - H_a)
        M_NS = peso * f_prime_mid + (1 - peso) * pendiente_secante
        
        H_nuevo = H_b - f_b / M_NS
        
        if abs(H_nuevo - H_b) < tol:
            return H_nuevo, i+1
        
        H_a, H_b = H_b, H_nuevo
    raise RuntimeError("El método de Newton-Secante no converge")

def newton_raphson_method_electroneutralidad(H_init, C1, pKa1, C2, pKa2, tol=1e-16, max_iter=50):
    H = H_init
    for i in range(max_iter):
        f_valor = electroneutralidad(H, C1, pKa1, C2, pKa2)
        f_derivada_valor = derivada_electroneutralidad(H, C1, pKa1, C2, pKa2)
        if f_derivada_valor == 0:
            print("El método de Newton-Raphson falla: derivada cero.")
            return np.nan
        H_new = H - f_valor / f_derivada_valor
        if abs(H_new - H) < tol:
            return H_new, i+1
        H = H_new
    raise RuntimeError("El método de Newton-Raphson no converge")

# Longitud de la columna
L = 10.0

# Puntos a lo largo de la columna
x_puntos = np.linspace(0, L, 10)  # 10 puntos a lo largo de la columna
valores_pH_Ns = []
valores_pH_NR = []
pKa1 = 4.756  # pKa del acético
pKa2 = 9.25  # pKa del amoniaco


# Listas de valores de C1 y C2
C1_values = np.linspace(1, 10, len(x_puntos))  # De 1 mM a 10 mM
C2_values = np.linspace(10, 1, len(x_puntos))  # De 10 mM a 1 mM

# Resolver pH en cada punto
for x, C1, C2 in zip(x_puntos, C1_values, C2_values):
    H0 = 1e-14  # Primera aproximación inicial de [H+]
    H1 = 1e-13  # Segunda aproximación inicial de [H+]
    H_init = 1e-14
    try:
        start_time = time.perf_counter()  # Inicio del cronómetro
        H_solucion, it = newton_secant_method_electroneutralidad(H0, H1, C1/1000, pKa1, C2/1000, pKa2,1e-16)
        end_time=time.perf_counter()
        
        it1.append(it)
        T1.append(end_time-start_time)

        if H_solucion <= 0:
            
            valores_pH_Ns.append(np.nan)  # Indicar un error si H+ es no positiva
        else:
            pH = -np.log10(H_solucion)  # Calcular pH
            valores_pH_Ns.append(pH)
    except ValueError:
        valores_pH_Ns.append(np.nan)  # Indicar un error si el método de Newton-Secante no converge

    # Newton-Raphson
    try:
        start_time=time.perf_counter()
        H_solucion, it = newton_raphson_method_electroneutralidad(H_init, C1/1000, pKa1, C2/1000, pKa2)
        end_time=time.perf_counter()

        it2.append(it)
        T2.append(end_time-start_time)

        if H_solucion <= 0:
            valores_pH_NR.append(np.nan)  # Indicar un error si H+ es no positiva
        else:
            pH = -np.log10(H_solucion)  # Calcular pH
            valores_pH_NR.append(pH)
    except RuntimeError:
        valores_pH_NR.append(np.nan)  # Indicar un error si el método de Newton-Raphson no converge


# Graficar los valores de pH
plt.plot(x_puntos, valores_pH_Ns, marker='o', label='Newton-Secante')
plt.xlabel('Posición en la columna (m)')
plt.ylabel('pH')
plt.title('Variación del pH a lo largo de la columna (Newton-Secante)')
plt.grid(True)
plt.show()

plt.plot(x_puntos, valores_pH_NR, marker='o', label='Newton-Raphson')
plt.xlabel('Posición en la columna (m)')
plt.ylabel('pH')
plt.title('Variación del pH a lo largo de la columna (Newton-Raphson)')
plt.grid(True)
plt.legend()
plt.show()

#-------------------------------------------------D-------------------------------------------------

#PRUEBA CON NEWTON-SECANTE

#Cantidad de iteraciones promedio para NS
it_promedio_NS= sum(it1)/10

#paso de segundo a microsegundo y trunco en 2 decimales dsps de la coma
suma_total_NS = sum(T1)
print(f"Para NEWTON-SECANTE el tiempo total es: {suma_total_NS} s (segundos) en {it_promedio_NS} iteraciones promedio")

# Crear un arreglo para el eje X que representa las pruebas (1 a 10)
ejecuciones = np.arange(1, 11)

# Graficar las iteraciones
plt.figure(figsize=(10, 6))  # Tamaño del gráfico

plt.bar(ejecuciones, it1, color='gray', alpha=0.7, label='Iteraciones')

plt.xlabel('Ejecuciones de prueba')
plt.ylabel('Cantidad de Iteraciones')
plt.title('Iteraciones realizadas en cada ejecución de prueba')
plt.xticks(ejecuciones)  # Asegurar que las etiquetas del eje X son enteros de 1 a 10
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.legend()
plt.show()

#PRUEBA CON NEWTON-RAPHSON

#Cantidad de iteraciones promedio para NS

it_promedio_NR= sum(it2)/10

#paso de segundo a microsegundo y trunco en 2 decimales dsps de la coma
suma_total_NR = sum(T2)
print(f"Para NEWTON-RAPHSON el tiempo total es: {suma_total_NR} s (segundos) en {it_promedio_NR} iteraciones promedio")

# Crear un arreglo para el eje X que representa las pruebas (1 a 10)
ejecuciones = np.arange(1, 11)

# Graficar las iteraciones
plt.figure(figsize=(10, 6))  # Tamaño del gráfico

plt.bar(ejecuciones, it2, color='gray', alpha=0.7, label='Iteraciones')

plt.xlabel('Ejecuciones de prueba')
plt.ylabel('Cantidad de Iteraciones')
plt.title('Iteraciones realizadas en cada ejecución de prueba')
plt.xticks(ejecuciones)  # Asegurar que las etiquetas del eje X son enteros de 1 a 10
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.legend()
plt.show()
#-------------------------------------------------E-------------------------------------------------
