import WebGUI
import HAL
import Frequency
import cv2

# Valores de velocidad lineal (V): PID
V_MAX = 15.0        # velocidad en la recta
V_MIN = 5.5         # velocidad mínima en la curva
KPV = 1.0           # P: cuánto frena por cada píxel de error
KIV = 0.0002        # I: freno extra si lleva un rato desviado
KDV = 0.03          # D: freno extra si el error crece rápido
INT_LIMIT_V = 2000    # el tope del acumulado de V 
ACCEL = 0.05        # el cuánto puede llegar a subir V por iteración

# Valores de velocidad de giro (W): PID
KP = 0.006
KI = 0.00002
KD = 0.10
INT_LIMIT = 1000    # el tope del acumulado de W
# si el error llega a bajar de este valor el acumulado vuelve al valor 0
RESET_ERROR = 10    

# Buscar la línea
# la velocidad lineal baja mientras busca la línea, avanza poco a poco 
V_SEARCH = 1.0
W_SEARCH = 2.0      # velocidad de giro fija mientras se busca la línea

# Memoria del PID de la velocidad de giro (W)
prev_error = 0      # error de la iteración anterior, para calcular la derivada
integral_v = 0      # sumatorio del error con signo

# Memoria del PID de la velocidad lineal (V)
prev_speed_error = 0    # error de velocidad de la iteración anterior 
integral_v = 0          # sumatorio del error de velocidad

# El estado del coche
last_direction = 1      # 1 = izquierda ; -1 = derecha
lost = False            # True mientras que no vea la línea
current_V = V_MIN       # la velocidad a la que realmente se envía al coche

while True:
    Frequency.tick()

    # La imagen y el filtro de rojo con HSV 
    image = HAL.getImage()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
           cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    h, w = mask.shape

    # La banda de filas por delante del coche
    # con 480 de alto: filas 180 a 405 
    top = int(h * 0.40)     # se quita lo muy lejano y el cielo
    bottom = int(h * 0.84)  # quitar lo pegado al coche
    mask[:top, :] = 0
    mask[bottom:, :] = 0

    # Centroide de la línea
    M = cv2.moments(mask)
    if M["m00"] > 0:
        error = w / 2 - M["m10"] / M["m00"]
        
        if lost:
            prev_error = error
            integral = 0
            integral_v = 0
            lost = False

        # PID de la velocidad del giro (W)
        if abs(error) < RESET_ERROR:
            # si el error es pequeño se borra el acumulado
            integral = 0
        else:
            integral = max(-INT_LIMIT, min(INT_LIMIT, integral + error))
        W = KP * error + KI * integral + KD * (error - prev_error)
        prev_error = error

        if abs(error) > 5:
            last_direction = 1 if error > 0 else -1

        # PID de la velocidad lineal (V), mira lo cercano y lo lejano
        far = mask[h // 2 : h // 2 + h // 8, :]
        M_far = cv2.moments(far)
        if M_far["m00"] > 0:
            error_far = w / 2 - M_far["m10"] / M_far["m00"]
            speed_error = max(abs(error), abs(error_far))

            if speed_error < RESET_ERROR:
                integral_v = 0
            else:
                integral_v = min(integral_v + speed_error, INT_LIMIT_V)
            
            brake = KPV * speed_error + KIV * integral_v + KDV * (speed_error - prev_speed_error)
            prev_speed_error = speed_error

            V = max(V_MIN, min(V_MAX, V_MAX - brake))
        else:
            V = V_MIN
    else:
        # Si no ve la línea que gire hacia donde la vio por última vez
        lost = True
        W = W_SEARCH * last_direction
        V = V_SEARCH
    
    # Frenar de golpe y acelerar poco a poco
    if V > current_V:
        current_V = min(V, current_V + ACCEL)
    else:
        current_V = V 

    HAL.setV(current_V)
    HAL.setW(W)
    WebGUI.showImage(mask)


