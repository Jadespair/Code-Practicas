import math
import WebGUI
import HAL
import Frequency

# Variables
VEL_MAX = 12.0        # velocidad lineal máxima
VEL_MIN = 2.5         # velocidad lineal mínima
W_MAX = 3.0           # velocidad angular máxima
K_GIRO = 2.0          # la ganancia proporcional, giro por cada radian de error
AMORTIGUACION = 1.5   # frena el giro si el ángulo se cambia rápido
INERCIA_W = 0.3       # es el peso del giro anterior
ALCANCE = 5.0         # el radio del láser, los obstáculos más lejanos que este valor no repelen
K_REPULSION = 8.0     # el peso de la repulsión frente a la atracción
TOPE_REPULSION = 2.0  # límite de la repulsión lateral para que no reaccione de más
DIST_FRENO = 3.0      # si llega a haber algo delante a menos de este valor, empieza a frenar
DIST_META = 3.0       # a esta distancia damos el subobjetivo como alcanzado


# Variables de la iteración anterior 
w_ant = 0.0
error_ant = 0.0

while True:
    Frequency.tick()

    objetivo = WebGUI.getNextTarget()     # el subobjetivo en coordenadas absolutas
    laser = HAL.getLaserData()
    coche = HAL.getPose3d()

    if len(laser.values) >= 180:
        # Si hay una lectura que no es válida se trata como si no hubiera 
        # obstáculo
        lecturas = [float('inf') if (d is None or math.isnan(d) or d <= 0) else d
                for d in laser.values[:180]]

        # Hacemos la distancia mínima en la zona frontal, unos +- 15 grados
        dist_delante = min(lecturas[75:106])

        # -- Fuerza atractiva
        # La diferencia entre la posición del subobjetivo y la del coche
        pos_obj = objetivo.getPose()
        dif_x = pos_obj.x - coche.x
        dif_y = pos_obj.y - coche.y

        # Rotamos la diferencia para pasarla a los ejes del coche
        # siendo x al frente e y a la izquierda
        atr_x = dif_x * math.cos(coche.yaw) + dif_y * math.sin(coche.yaw)
        atr_y = dif_x * math.sin(coche.yaw) + dif_y * math.cos(coche.yaw)

        # Normalizamos la dirección para que no dependa de lo lejos que esté el subobjetivo
        dist_obj = math.hypot(atr_x, atr_y)
        if dist_obj > 0:
            atr_x, atr_y = atr_x / dist_obj, atr_y / dist_obj

        
        # -- Fuerza repulsiva
        # Cada rayo del láser que detecte algo cerca empuja al coche en sentido contrario
        # La intensidad es de 1 / distancia^2, para que lo cercano pese mucho más que lo lejano
        rep_x = 0.0
        rep_y = 0.0
        for i, d in enumerate(lecturas):
            if d < ALCANCE:
                ang_rayo = math.radians(i - 90)     # 0 = frente; positivo = izq
                intensidad = 1.0 / max(d, 0.15) ** 2    # el máx evita dividir por -0
                rep_x -= intensidad * math.cos(ang_rayo)
                rep_y -= intensidad * math.sin(ang_rayo)

        # Escalamos la suma y limitamos el componente lateral
        rep_x = K_REPULSION * rep_x / 60
        rep_y = K_REPULSION * rep_y / 60
        rep_y = max(-TOPE_REPULSION, min(TOPE_REPULSION, rep_y))


        # -- Fuerza resultante
        # La repulsión hace que el avance, el cual es marcado por la atracción
        # La repulsión solo influye en el giro, el eje y
        fuerza_x = max(atr_x, 0.6)
        fuerza_y = atr_y + rep_y

        # Si hay un obstáculo justo delante suya, las fuerzas laterales se anulan
        # así que se gira hacia el lado donde haya más hueco
        if dist_delante < 1.0 and abs(fuerza_y) < 0.05:
            h_izq = sum(min(d, ALCANCE) for d in lecturas[110:150])
            h_der = sum(min(d, ALCANCE) for d in lecturas[30:70])
            fuerza_y = 0.5 if h_izq > h_der else -0.5

        # El ángulo de la fuerza resultante respecto al frente del coche
        angulo = math.atan2(fuerza_y, fuerza_x)


        # -- Velocidad angular W
        w = K_GIRO * angulo + AMORTIGUACION * (angulo - error_ant)
        w = max(-W_MAX, min(W_MAX, w))
        w = INERCIA_W * w_ant + (1 - INERCIA_W) * w
        error_ant = angulo
        w_ant = w


        # -- Velocidad lineal, baja al girar y cuanto más cerca haya algo delante
        por_giro = 1.0 - 0.6 * min(abs(angulo) / (math.pi / 2), 1.0)
        por_frente = min(1.0, dist_delante / DIST_FRENO) ** 2
        v = max(VEL_MIN, VEL_MAX * por_giro * por_frente)

        HAL.setV(v)
        HAL.setW(w)

        # Dibujamos las tres fuerzas en la interfaz, la atractiva, la repulsiva y la resultante
        WebGUI.showForces([atr_x, atr_y], [rep_x, rep_y], [fuerza_x, fuerza_y])

        # Si el subobjetivo ha sido alcanzado se marca sin parar el coche y se pasa al siguiente
        if dist_obj < DIST_META:
            objetivo.setReached(True)


