import WebGUI
import HAL
import Frequency
import numpy as np
import cv2
import heapq
import math

# Valores de constantes
MAP_URL = '/resources/exercises/global_navigation/images/cityLargenBin.png'
VALOR_CARRETERA = 255  # valor de los píxeles de carretera en la imagen
M_CELDA = 1.25     	   # metros por celda (500 m / 400 celdas)

# El engordado de los obstáculos y penalización por la cercanía
N_CELDAS = 4           # celdas que se engordan los obstáculos
D_PEN = 12             # hasta qué distancia de la pared hay penali
PESO_PEN = 0.7         # lo fuerte que es la penalización (< 1)

# La ruta local por delante del coche
PASOS_LEJOS = 36       # cuántas celdas por delante se calcula la ruta
D_SEG = 5.0            # distancia a la pared hasta la que se aparta la ruta 
ALEJAR_PASOS = 3       # máximo que se aparta cada punto
LOOK_MIN = 6.0		   # distancia de mira con el coche parado
LOOK_MAX = 20.0		   # distancia de mira máxima
LOOK_K = 0.7		   # los metros de mira extra por cada unidad de velocidad
LOOK_GIRO = 12.0 	   # distancia de mira mientras gira sobre sí mismo

# Velocidad lineal
V_MAX = 25.0		   # velocidad máxima en rectas
A_LAT = 6.5            # aceleración lateral admisible en curvas
A_FRENO = 6.0          # frenado
V_FIN = 2.0            # velocidad con la que llega al destino
K_V = 0.8              # frenada extra cuando el coche está mal orientado
DV_SUBE = 0.3          # arranque progresivo, aumento máximo de V por iteración
V_ESPACIO_MIN = 5.0    # velocidad máxima pegado a la pared
K_ESPACIO = 4.0        # m/s extra por cada celda de espacio libre

# Velocidad angular
K_W = 1.0			   # la ganancia
V_REF = 15.0           # por encima de esta V la ganancia de W baja
K_D = 0.4			   # amortiguar las oscilaciones
ALFA_W = 0.5           # filtro de W, para girar más suave
W_MAX = 2.5

# Girar sobre sí mismo
ERROR_GIRO = 1.2       # error (rad) a partir del cual para y gira en el sitio
ERROR_FIN_GIRO = 0.15  # error (rad) por debajo del cual vuelve a avanzar
W_GIRO = 2.0		   # velocidad angular máxima al girar en el sitio
W_GIRO_MIN = 0.4	   # velocidad angular mínima, para que acabe el giro
K_GIRO = 2.0		   # W al girar es igual a K_GIRO * error, entre el mínimo y el máximo

DIST_LLEGADA = 3       # celdas al destino para que se dé por alcanzado

# Los 8 vecinos de una celda. 
# Cambio de fila, cambio de columna, logintud del paso.
# Los rectos van primero, si dos vecinos empatan, gana el recto.
VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


# Utilidades
def limitar(x, minimo, maximo):
    return max(minimo, min(maximo, x))

def normalizar(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# convertimos las coordenadas (x, y) a (fila, columna)
def mundo_a_celda(punto):
    col, fila = WebGUI.worldToGrid(punto)
    return fila, col


# Construimos el campo
def engordar(mapa):
    obstaculos = (mapa != VALOR_CARRETERA).astype(np.uint8)
    tam = 2 * N_CELDAS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))
    return np.where(cv2.dilate(obstaculos, kernel) == 0, 255, 0).astype(np.uint8)

# usamos el algoritmo de Dijkstra para propagar
def propagar(libre, destino):
    filas, cols = libre.shape
    ancho = cols + 2             # borde de 1 celda prohibida
    libre_l = np.pad(libre == 255, 1, constant_values=False).ravel().tolist()
    # Cada vecino como un desplazamiento en la lista plana
    vecinos = [(df * ancho + dc, largo) for df, dc, largo in VECINOS]

    campo = [float('inf')] * len(libre_l)
    inicio = (destino[0] + 1) * ancho + (destino[1] + 1)
    campo[inicio] = 0.0         # el destino vale 0
    cola = [(0.0, inicio)]      # siempre se expande la celda más cercana
    while cola:
        coste, i = heapq.heappop(cola)
        if coste > campo[i]:   
            continue
        for d, largo in vecinos:
            j = i + d
            if libre_l[j] and coste + largo < campo[j]:
                campo[j] = coste + largo
                heapq.heappush(cola, (campo[j], j))
    return np.array(campo, dtype=np.float32).reshape(filas + 2, ancho)[1:-1, 1:-1]

# hacemos el camino bajando por el campo, donde en cada paso elige el vecino con mayor bajada por unidad de longitud
def ruta_gradiente(total, f, c, max_pasos):
    filas, cols = total.shape
    ruta = [(f, c)]
    for _ in range(max_pasos):
        mejor, mejor_pend = None, 0.0
        for df, dc, largo in VECINOS:
            nf, nc = f + df, c + dc
            if 0 <= nf < filas and 0 <= nc < cols and total[nf, nc] < total[f, c]:
                pendiente = (total[f, c] - total[nf, nc]) / largo
                if pendiente > mejor_pend + 1e-9:
                    mejor, mejor_pend = (nf, nc), pendiente
        if mejor is None:      # ningún vecino baja, se ha alcanzado el destino
            break
        f, c = mejor
        ruta.append(mejor)
    return ruta

# buscamos la celda más cercana
def celda_alcanzable_cercana(total, f, c, radio=8):
    f0, c0 = max(0, f - radio), max(0, c - radio)
    filas, cols = np.nonzero(np.isfinite(total[f0:f + radio + 1, c0:c + radio + 1]))
    if len(filas) == 0:
        return None
    i = np.argmin((filas + f0 - f) ** 2 + (cols + c0 - c) ** 2)
    return int(filas[i] + f0), int(cols[i] + c0)

# dibujamos el campo, donde el destino se ve más claro y la ruta en blanco
def campo_a_imagen(campo, ruta):
    img = np.zeros(campo.shape, dtype=np.uint8)
    validas = np.isfinite(campo)
    img[validas] = 40 + 160 * (1 - campo[validas] / campo[validas].max())
    for rf, rc in ruta:
        img[max(0, rf - 1):rf + 2, max(0, rc - 1):rc + 2] = 255
    return img


# Ruta a V y W
# quitamos la oscilación que pueda causar la rejilla
def suavizar(ruta, ventana=7):
    pts = np.array(ruta, dtype=float)
    m = ventana // 2
    return np.array([pts[max(0, i - m):i + m + 1].mean(axis=0) for i in range(len(pts))])

# hacemos que el coche se pueda abrir haciendo la curva antes de la esquina y salga recta eligiendo el punto de mira
def alejar_de_paredes(pts):
    filas, cols = dist_pared.shape
    salida = pts.copy()
    for k in range(len(salida)):
        p = salida[k].copy()
        for _ in range(ALEJAR_PASOS):
            r, q = int(round(p[0])), int(round(p[1]))
            if not (0 <= r < filas and 0 <= q < cols) or dist_pared[r, q] >= D_SEG:
                break            # ya está lo bastante lejos
            g = np.array([grad_f[r, q], grad_c[r, q]])  # hacia dónde crece la distancia
            if np.linalg.norm(g) < 0.3:                 # eje de la calle
                break
            nuevo = p + g / np.linalg.norm(g)           # un paso de 1 celda
            rn, qn = int(round(nuevo[0])), int(round(nuevo[1]))
            if not (0 <= rn < filas and 0 <= qn < cols) or dist_pared[rn, qn] <= dist_pared[r, q]:
                break       # no se gana distancia
            p = nuevo
        salida[k] = p
    return salida


def curvatura(a, b, c):
    cruz = abs((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0])
    den = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
    return 2.0 * cruz / den if den > 1e-9 else 0.0

# la distancia de un punto de la ruta a la zona prohibida
def holgura(p):
    return dist_pared[int(limitar(round(p[0]), 0, 399)), int(limitar(round(p[1]), 0, 399))]

# ponemos una velocidad máxima para poder frenar a tiempo antes de cada curva y tramo estrecho
def velocidad_permitida(pts, llega_destino):
    n = len(pts)
    s = 0.0                               # distancia recorrida de la ruta (m)
    v_perm = min(V_MAX, V_ESPACIO_MIN + K_ESPACIO * holgura(pts[0]))
    for i in range(1, n):
        s += np.linalg.norm(pts[i] - pts[i - 1]) * M_CELDA
        k = curvatura(pts[max(0, i - 4)], pts[i], pts[min(n - 1, i + 4)]) / M_CELDA        
        v_i = V_MAX if k < 1e-6 else min(V_MAX, math.sqrt(A_LAT / k))   # el límite por la curva 
        v_i = min(v_i, V_ESPACIO_MIN + K_ESPACIO * holgura(pts[i]))     # el límite por el espacio
        v_perm = min(v_perm, math.sqrt(v_i ** 2 + 2 * A_FRENO * s))     # con tiempo de frenar
    if llega_destino:
        v_perm = min(v_perm, math.sqrt(V_FIN ** 2 + 2 * A_FRENO * s))
    return v_perm

# El primer punto de la ruta que está a dist_m medidos a lo largo de la ruta
def punto_mira(pts, dist_m):
    s = 0.0
    for i in range(1, len(pts)):
        s += np.linalg.norm(pts[i] - pts[i - 1]) * M_CELDA
        if s >= dist_m:
            return pts[i]
    return pts[-1]


mapa = WebGUI.getMap(MAP_URL)
libre = engordar(mapa)          

# La distancia de cada celda libre a la zona prohibida y, hacia dónde crece
dist_pared = cv2.distanceTransform(libre, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
grad_f, grad_c = np.gradient(dist_pared)               

# Subcampo de penalización
pen = (np.clip(D_PEN - dist_pared, 0, None) * PESO_PEN).astype(np.float32)  

imagen = libre          # lo que se dibuja hasta que haya destino elegido

destino_anterior, total, celda_destino = None, None, None
llegado = False         # Se vuelve True cuando ya está en el destino
girando = False         # Se vuelve True mientras gira sobre sí mismo

error_prev, W_prev, V_prev = 0.0, 0.0, 0.0      # valores de la iteración anterior

while True:
    Frequency.tick()
    destino = WebGUI.getTargetPose()

    # Cuando se marca un destino nuevo, calculamos el campo
    if destino is not None and destino != destino_anterior:
        destino_anterior = destino
        fila, col = mundo_a_celda(destino)
        if not (0 <= fila < 400 and 0 <= col < 400) or libre[fila, col] != 255:
            print("Destino fuera del mapa")
        else:
            total = propagar(libre, (fila, col)) + pen  # campo = propagación + penalización
            celda_destino = (fila, col)
            llegado = False
            girando = False
            error_prev, W_prev, V_prev = 0.0, 0.0, 0.0

            # Dibujamos la ruta completa que se sugiere desde el coche
            p = HAL.getPose3d()
            fc, cc = mundo_a_celda([p.x, p.y])
            imagen = campo_a_imagen(total, ruta_gradiente(total, fc, cc, 2000))

    # Pilotar el coche siguiendo la gradiante del campo
    if total is None or llegado:
        HAL.setV(0)         # si no hay destino o ya ha llegado = parado
        HAL.setW(0)
    else:
        # Dónde se encuentra el coche y si ha llegado al destino
        pose = HAL.getPose3d()
        f0, c0 = mundo_a_celda([pose.x, pose.y])

        if math.hypot(f0 - celda_destino[0], c0 - celda_destino[1]) < DIST_LLEGADA:
            llegado = True
            HAL.setV(0)
            HAL.setW(0)
            print("Destino alcanzado")
        else:
            # Ruta local por delante del coche
            if np.isfinite(total[f0, c0]):
                ruta_loc = ruta_gradiente(total, f0, c0, PASOS_LEJOS)
            else:       
                # si el coche pisa la zona engordada se empieza a ir a la celda válida más cercana
                cerca = celda_alcanzable_cercana(total, f0, c0) or (f0, c0)
                ruta_loc = [(f0, c0)] + ruta_gradiente(total, cerca[0], cerca[1], PASOS_LEJOS)
            pts = suavizar(ruta_loc)

            # Punto de mira: más lejos cuanto más rápido va. Se elige sobre la ruta apartada de las paredes
            look = min(LOOK_MAX, LOOK_MIN + LOOK_K * V_prev)
            if girando:
                look = max(look, LOOK_GIRO)
            f1, c1 = punto_mira(alejar_de_paredes(pts), look)
            if math.hypot(f1 - f0, c1 - c0) < 0.5:      # el punto de mira coincide con el coche
                f1, c1 = celda_destino

            # Error de ángulo hacia el punto de mira (fila = x, columna = y)
            error = normalizar(math.atan2(c1 - c0, f1 - f0) - pose.yaw)

            # Vuelta a sí mismo
            if abs(error) > ERROR_GIRO:
                girando = True
            elif abs(error) < ERROR_FIN_GIRO:
                girando = False

            # Calculamos V y W
            if girando:     
                V = 0.0
                W = math.copysign(limitar(K_GIRO * abs(error), W_GIRO_MIN, W_GIRO), error)
                error_prev, W_prev = 0.0, 0.0   # sin inercia del filtro al salir del giro
            else:
                # W: PD sobre el error + filtro
                # la ganancia baja por encima de V_REF
                k_w = K_W * V_REF / max(V_prev, V_REF)
                W_raw = limitar(k_w * error + K_D * (error - error_prev), -W_MAX, W_MAX)
                W = ALFA_W * W_raw + (1 - ALFA_W) * W_prev

                # V: la que permite la curva y el espacio de la ruta 
                v_perm = velocidad_permitida(pts, len(ruta_loc) - 1 < PASOS_LEJOS)
                V = min(v_perm / (1 + K_V * abs(error)), V_prev + DV_SUBE)
                error_prev, W_prev = error, W

            # Enviamos las órdenes al coche
            V_prev = V
            HAL.setV(V)
            HAL.setW(W)

    WebGUI.showNumpy(imagen)        # Dibujamos el campo y su ruta
