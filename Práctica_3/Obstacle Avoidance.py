import math
import WebGUI
import HAL
import Frequency

# ---------- Parámetros ----------
VEL_MAX = 8.5        # velocidad lineal máxima (m/s), la que lleva en recta libre
VEL_MIN = 2.5        # velocidad lineal mínima: nunca baja de aquí para no perder tiempo
GIRO_MAX = 3.0       # velocidad angular máxima (rad/s)
GANANCIA_GIRO = 2.0  # cuánto gira por cada radián de error entre el rumbo y la fuerza resultante
SUAVIZADO = 0.3      # peso del giro anterior: 0 = sin suavizar, 1 = el giro nunca cambia
ALCANCE = 3.0        # más allá de esta distancia (m) un obstáculo no repele
PESO_REP = 8.0       # peso de la repulsión frente a la atracción (que vale 1)
DIST_FRENO = 3.0     # a menos distancia frontal que esta (m) el coche empieza a frenar
REP_MAX = 2.0        # tope de la repulsión lateral, para que no reaccione de forma exagerada
FILTRO_REP = 0.5     # suaviza la repulsión entre iteraciones (0 = sin filtrar)
AMORTIGUACION = 1.5  # término derivativo del giro: frena el giro si el ángulo ya cambia rápido
AVANCE_MIN = 0.6     # mínimo de la componente de avance; más alto = ángulos menos bruscos
RADIO_META = 3.0     # a esta distancia (m) se da el subobjetivo por alcanzado


# ---------- Láser ----------
def leer_distancias(laser):
    # Devuelve las 180 distancias del láser. Las lecturas inválidas
    # (None, NaN o <= 0) se tratan como "sin obstáculo" (infinito)
    dist = []
    for i in range(180):
        d = laser.values[i]
        if d is None or math.isnan(d) or d <= 0:
            d = float('inf')
        dist.append(d)
    return dist


# ---------- Fuerza atractiva (vector unitario hacia el subobjetivo) ----------
def a_relativas(x, y, rx, ry, yaw):
    # Pasa un punto de coordenadas absolutas (mapa) a coordenadas del coche:
    # primero se resta la posición del coche y luego se rota según su orientación.
    # En el sistema del coche: x = hacia delante, y = hacia la izquierda
    dx, dy = x - rx, y - ry
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


def atraccion(target, pose):
    # Devuelve la dirección hacia el subobjetivo (módulo 1) y la distancia a él.
    # Se normaliza para que la fuerza no dependa de lo lejos que esté el punto
    p = target.getPose()
    x, y = a_relativas(p.x, p.y, pose.x, pose.y, pose.yaw)
    d = math.hypot(x, y)
    if d == 0:
        return 0.0, 0.0, 0.0
    return x / d, y / d, d


# ---------- Fuerza repulsiva (1/d² de cada rayo) ----------
def repulsion(dist):
    # Cada rayo con obstáculo cercano empuja al coche en sentido contrario a su dirección.
    # Se usa 1/d² para que lo cercano repela mucho más que lo lejano
    fx = fy = 0.0
    for i in range(180):
        d = dist[i]
        if d < ALCANCE:
            d = max(d, 0.15)                # evita dividir por casi cero
            ang = math.radians(i - 90)      # 0 = frente, positivo = izquierda
            m = 1.0 / (d * d)               # módulo de la repulsión de este rayo
            fx -= m * math.cos(ang)         # el signo menos: empuja alejándose del obstáculo
            fy -= m * math.sin(ang)
    # Se divide entre 60 para escalar la suma y se pondera con PESO_REP
    return PESO_REP * fx / 60, PESO_REP * fy / 60


# ---------- Bucle principal ----------
w_ant = 0.0          # velocidad angular de la iteración anterior (para suavizar)
ang_ant = 0.0        # ángulo de la fuerza en la iteración anterior (para la derivada)
rep_filtrada = 0.0   # repulsión lateral filtrada
lado = 1             # lado preferido si hay indecisión (1 = izquierda, -1 = derecha)

while True:
    Frequency.tick()

    target = WebGUI.getNextTarget()     # siguiente subobjetivo (coordenadas absolutas)
    laser = HAL.getLaserData()
    pose = HAL.getPose3d()

    if len(laser.values) >= 180:
        dist = leer_distancias(laser)
        d_frente = min(dist[75:106])    # distancia mínima en ±15° al frente

        # 1. Fuerzas
        ax, ay, d_meta = atraccion(target, pose)
        rx, ry = repulsion(dist)

        # 2. Combinación.
        # La repulsión suele apuntar hacia atrás (el láser solo mira al frente),
        # así que no se suma a la componente de avance: esta la marca solo la
        # atracción (con un mínimo). La repulsión solo influye en el giro (eje y)
        ry = max(-REP_MAX, min(REP_MAX, ry))                      # tope
        rep_filtrada = FILTRO_REP * rep_filtrada + (1 - FILTRO_REP) * ry   # filtro paso bajo
        fx = max(ax, AVANCE_MIN)
        fy = ay + rep_filtrada

        # 3. Indecisión: si hay algo delante y las fuerzas laterales se anulan,
        # el coche iría de frente al obstáculo. Se fuerza un giro hacia el lado con más hueco
        if d_frente < 1.0 and abs(fy) < 0.05:
            izq = sum(min(d, ALCANCE) for d in dist[110:150])
            der = sum(min(d, ALCANCE) for d in dist[30:70])
            # histéresis: solo cambia de lado si la diferencia es clara, para no oscilar
            if izq > der + 3.0:
                lado = 1
            elif der > izq + 3.0:
                lado = -1
            fy = 0.5 * lado

        # 4. Dirección de la fuerza resultante respecto al frente del coche
        angulo = math.atan2(fy, fx)

        # 5. Velocidad angular: proporcional al ángulo + término derivativo,
        # limitada y suavizada con el giro anterior para evitar zigzag
        w = GANANCIA_GIRO * angulo + AMORTIGUACION * (angulo - ang_ant)
        w = max(-GIRO_MAX, min(GIRO_MAX, w))
        ang_ant = angulo
        w = SUAVIZADO * w_ant + (1 - SUAVIZADO) * w
        w_ant = w

        # 6. Velocidad lineal: baja si hay que girar mucho o hay algo delante
        por_giro = 1.0 - 0.6 * min(abs(angulo) / (math.pi / 2), 1.0)
        por_frente = min(1.0, d_frente / DIST_FRENO) ** 2
        v = max(VEL_MIN, VEL_MAX * por_giro * por_frente)

        HAL.setV(v)
        HAL.setW(w)
        # Verde: atracción, rojo: repulsión, negro: resultante
        WebGUI.showForces([ax, ay], [rx, rep_filtrada], [fx, fy])

        # 7. Subobjetivo alcanzado: se marca sin parar el coche y se pasa al siguiente
        if d_meta < RADIO_META:
            target.setReached(True)
