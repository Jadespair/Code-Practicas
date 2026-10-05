import WebGUI
import HAL
import Frequency
import random 
import time

estado = "ESPIRAL"              # el estado en el que comienza el robot
inicio = time.time()            # momento en el que comenzó el estado actual
tiempo_giro = 0                 
direccion = 1                   # sentido del giro, 1 = izquierda ; -1 = derecha

while True:
    Frequency.tick(50)          # Regula el bucle a unas 50 vueltas por segundo

    # Leemos el láser 
    laser = HAL.getLaserData()

    # Si todavía no hay 180 datos (1 por cada grado), se para el robot y se espera a la siguiente vuelta 
    if len(laser.values) < 180:
        HAL.setV(0)
        HAL.setW(0)
        continue   

    # El 0 es el lado derecho, el 90 es el frente y el 179 es el lado izquierdo
    centro = 5                  # objeto más cercano de frente
    lado_izq = 5                # objeto más cercano a la izquierda
    lado_der = 5                # objeto más cercano a la derecha
    izquierda = 5               # la suma de distancias de la izquierda, el espacio libre
    derecha = 5                 # la suma de distancias de la derecha, el espacio libre

    # Recorremos los rayos del láser por cada zona, solo se acepta una lectura 
    for i in range(60, 121):    # zona central (30 grados a cada lado)  
        valor = laser.values[i]
        if valor >= 0 and valor < 5:
            centro = min(centro, valor)

    for i in range(140, 180):   # zona lateral izquierda 
        valor = laser.values[i]
        if valor >= 0 and valor < 5:
            lado_izq = min(lado_izq, valor)

    for i in range(0, 40):      # zona lateral derecha
        valor = laser.values[i]
        if valor >= 0 and valor < 5:
            lado_der = min(lado_der, valor)

    for i in range(135, 180):   # el espacio total de la izquierda
        valor = laser.values[i]
        if valor >= 0 and valor < 5:
            izquierda = izquierda + valor
        else:
            izquierda = izquierda + 5   

    for i in range(0, 45):     # el espacio total de la derecha
        valor = laser.values[i]
        if valor >= 0 and valor < 5:
            derecha = derecha + valor
        else:
            derecha = derecha + 5

    # Si hay un obstáculo a menos de 0'40 metros por delante
    # o a menos de 0'22 metros por cualquiera de los dos lados
    obstaculo = centro < 0.40 or lado_izq < 0.22 or lado_der < 0.22

    # EL tiempo que lleva el robot en el estado actual
    tiempo = time.time() - inicio

    # Máquina de estados
    # Código de la espiral 
    if estado == "ESPIRAL":
        v = 0.5                 # velocidad de avance fija
        r = 0.4 + 0.2 * tiempo  # el radio aumenta 0'2 metros por segundo
        HAL.setV(v)
        HAL.setW(v / r)         # gira cada vez más despacio, ya que el círculo se abre
        if obstaculo:
            inicio = time.time()     
            estado = "RETROCEDER"
        elif tiempo > 8:        # si el radio es muy grande, mejor ir recto 
            inicio = time.time()
            estado = "AVANZAR"

    # Código de avanzar
    elif estado == "AVANZAR":
        HAL.setV(0.6)
        HAL.setW(0)
        if obstaculo:
            inicio = time.time()
            estado = "RETROCEDER"

    # Código de retroceder
    elif estado == "RETROCEDER":
        HAL.setV(-0.15)         # velocidad negativa, hacia atrás
        HAL.setW(0)
        if tiempo > 0.2:
            if izquierda > derecha:  # si hay más espacio libre a la izquierda, giramos a la izquierda
                direccion = 1
            else:                     # si hay más espacio libre a la derecha, giramos a la derecha       
                direccion = -1
            tiempo_giro = random.uniform(1.0, 2.5)  # giramos un tiempo aleatorio  
            inicio = time.time()
            estado = "GIRAR"

    # Código de girar
    elif estado == "GIRAR":
        HAL.setV(0)
        HAL.setW(direccion * 1.0)  
        if tiempo > tiempo_giro:
            inicio = time.time()
            # Hace movimiento en espiral si hay espacio libre, si no, avanza recto
            if centro > 1.5 and lado_izq > 1.0 and lado_der > 1.0:
                estado = "ESPIRAL"
            else:
                estado = "AVANZAR"

