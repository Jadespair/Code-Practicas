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
M_CELDA = 1.25         # metros por celda (500 m / 400 celdas)

# El engordado de los obstáculos y penalización por la cercanía
N_CELDAS = 4           # celdas que se engordan los obstáculos
D_PEN = 12             # hasta qué distancia de la pared hay penalización
PESO_PEN = 0.7         # lo fuerte que es la penalización (< 1)

# Dibujo del mapa
D_VIS = 6.0            # celdas desde la pared hasta que el fondo se ve oscuro
BRILLO_FONDO = 25      # brillo en el centro de la calle (0-255)
BRILLO_BORDE = 255     # brillo pegado a la pared (0-255), antes 130

# La ruta local por delante del coche
PASOS_LEJOS = 36       # cuántas celdas por delante se calcula la ruta
D_SEG = 5.0            # distancia a la pared hasta la que se aparta la ruta
ALEJAR_PASOS = 3       # máximo que se aparta cada punto
LOOK_MIN = 6.0         # distancia de mira con el coche parado
LOOK_MAX = 20.0        # distancia de mira máxima
LOOK_K = 0.7           # los metros de mira extra por cada unidad de velocidad
LOOK_GIRO = 12.0       # distancia de mira mientras gira sobre sí mismo

# Velocidad lineal
V_MAX = 25.0           # velocidad máxima en rectas
A_LAT = 6.5            # aceleración lateral admisible en curvas
A_FRENO = 6.0          # frenado
V_FIN = 2.0            # velocidad con la que llega al destino
K_V = 0.8              # frenada extra cuando el coche está mal orientado
DV_SUBE = 0.3          # arranque progresivo, aumento máximo de V por iteración
V_ESPACIO_MIN = 5.0    # velocidad máxima pegado a la pared
K_ESPACIO = 4.0        # m/s extra por cada celda de espacio libre

# Velocidad angular
K_W = 1.0              # la ganancia
V_REF = 15.0           # por encima de esta V la ganancia de W baja
K_D = 0.4              # amortiguar las oscilaciones
ALFA_W = 0.5           # filtro de W, para girar más suave
W_MAX = 2.5

# Girar sobre sí mismo
ERROR_GIRO = 1.2       # error (rad) a partir del cual para y gira en el sitio
ERROR_FIN_GIRO = 0.15  # error (rad) por debajo del cual vuelve a avanzar
W_GIRO = 2.0           # velocidad angular máxima al girar en el sitio
W_GIRO_MIN = 0.4       # velocidad angular mínima, para que acabe el giro
K_GIRO = 2.0           # W al girar es igual a K_GIRO * error, entre el mínimo y el máximo

DIST_LLEGADA = 3       # celdas al destino para que se dé por alcanzado

# Los 8 vecinos de una celda.
# Cambio de fila, cambio de columna, longitud del paso.
# Los rectos van primero, si dos vecinos empatan, gana el recto.
VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


# Utilidades
def limitar(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def normalizar_angulo(ang):
    return (ang + math.pi) % (2 * math.pi) - math.pi


# convertimos las coordenadas (x, y) a (fila, columna)
def mundo_a_celda(punto):
    col, fila = WebGUI.worldToGrid(punto)
    return fila, col


# Construimos el campo
def engordar_obstaculos(mapa):
    obst = (mapa != VALOR_CARRETERA).astype(np.uint8)
    tam = 2 * N_CELDAS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))
    return np.where(cv2.dilate(obst, kernel) == 0, 255, 0).astype(np.uint8)


# usamos el algoritmo de Dijkstra para propagar
def propagar_desde_destino(libre, destino):
    n_filas, n_cols = libre.shape
    ancho = n_cols + 2     # borde de 1 celda prohibida
    libre_l = np.pad(libre == 255, 1, constant_values=False).ravel().tolist()
    # Cada vecino como un desplazamiento en la lista plana
    vec_l = [(d_f * ancho + d_c, paso) for d_f, d_c, paso in VECINOS]

    campo = [float('inf')] * len(libre_l)
    i_dest = (destino[0] + 1) * ancho + (destino[1] + 1)
    campo[i_dest] = 0.0       # el destino vale 0
    cola = [(0.0, i_dest)]    # siempre se expande la celda más cercana
    while cola:
        coste, i = heapq.heappop(cola)
        if coste > campo[i]:
            continue
        for desp, paso in vec_l:
            j = i + desp
            if libre_l[j] and coste + paso < campo[j]:
                campo[j] = coste + paso
                heapq.heappush(cola, (campo[j], j))
    return np.array(campo, dtype=np.float32).reshape(n_filas + 2, ancho)[1:-1, 1:-1]


# hacemos el camino bajando por el campo, donde en cada paso elige el vecino con mayor bajada por unidad de longitud
def ruta_por_gradiente(total, fila, col, max_pasos):
    n_filas, n_cols = total.shape
    ruta = [(fila, col)]
    for _ in range(max_pasos):
        mejor, mejor_p = None, 0.0
        for d_f, d_c, paso in VECINOS:
            nf, nc = fila + d_f, col + d_c
            if (0 <= nf < n_filas and 0 <= nc < n_cols
                    and total[nf, nc] < total[fila, col]):
                pend = (total[fila, col] - total[nf, nc]) / paso
                if pend > mejor_p + 1e-9:
                    mejor, mejor_p = (nf, nc), pend
        if mejor is None:      # ningún vecino baja, se ha alcanzado el destino
            break
        fila, col = mejor
        ruta.append(mejor)
    return ruta


# buscamos la celda más cercana
def celda_alcanzable_cercana(total, fila, col, radio=8):
    f_ini, c_ini = max(0, fila - radio), max(0, col - radio)
    ventana = total[f_ini:fila + radio + 1, c_ini:col + radio + 1]
    fs, cs = np.nonzero(np.isfinite(ventana))
    if len(fs) == 0:
        return None
    i = np.argmin((fs + f_ini - fila) ** 2 + (cs + c_ini - col) ** 2)
    return int(fs[i] + f_ini), int(cs[i] + c_ini)


# Mapa de fondo de UN canal, calculado sobre el mapa ORIGINAL (calles con su anchura real).
# Obstáculos en negro; en la carretera, más claro cuanto más cerca de la pared:
# BRILLO_BORDE junto a la pared, BRILLO_FONDO en el centro de la calle.
def imagen_del_mapa(mapa):
    carretera = (mapa == VALOR_CARRETERA)
    dist_real = cv2.distanceTransform(carretera.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cerca = np.clip(1.0 - dist_real / D_VIS, 0.0, 1.0)     # 1 junto a la pared, 0 lejos
    img = np.zeros(mapa.shape, dtype=np.uint8)
    img[carretera] = (BRILLO_FONDO + (BRILLO_BORDE - BRILLO_FONDO) * cerca[carretera]).astype(np.uint8)
    return img


# copia el mapa de fondo y dibuja la ruta como línea blanca con contorno negro
def dibujar_ruta(img_base, ruta):
    img = img_base.copy()
    if len(ruta) >= 2:
        linea = np.array([(c, f) for f, c in ruta], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [linea], False, 0, 7)       # contorno negro
        cv2.polylines(img, [linea], False, 255, 3)     # línea blanca
    return img


# Ruta a velocidad lineal y angular
# quitamos la oscilación que pueda causar la rejilla
def suavizar_ruta(ruta, ventana=7):
    pts = np.array(ruta, dtype=float)
    mitad = ventana // 2
    return np.array([pts[max(0, i - mitad):i + mitad + 1].mean(axis=0)
                     for i in range(len(pts))])


# hacemos que el coche se pueda abrir haciendo la curva antes de la esquina y salga recto eligiendo el punto de mira
def alejar_de_paredes(pts):
    n_filas, n_cols = dist_pared.shape
    pts_a = pts.copy()
    for i in range(len(pts_a)):
        p = pts_a[i].copy()
        for _ in range(ALEJAR_PASOS):
            pf, pc = int(round(p[0])), int(round(p[1]))
            if (not (0 <= pf < n_filas and 0 <= pc < n_cols)
                    or dist_pared[pf, pc] >= D_SEG):
                break      # ya está lo bastante lejos
            # hacia dónde crece la distancia a la pared
            direc = np.array([grad_f[pf, pc], grad_c[pf, pc]])
            if np.linalg.norm(direc) < 0.3:      # eje de la calle
                break
            p_nuevo = p + direc / np.linalg.norm(direc)     # un paso de 1 celda
            nf, nc = int(round(p_nuevo[0])), int(round(p_nuevo[1]))
            if (not (0 <= nf < n_filas and 0 <= nc < n_cols)
                    or dist_pared[nf, nc] <= dist_pared[pf, pc]):
                break      # no se gana distancia
            p = p_nuevo
        pts_a[i] = p
    return pts_a


def curvatura_tres_puntos(p1, p2, p3):
    cruz = abs((p2 - p1)[0] * (p3 - p2)[1] - (p2 - p1)[1] * (p3 - p2)[0])
    den = np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p2) * np.linalg.norm(p3 - p1)
    return 2.0 * cruz / den if den > 1e-9 else 0.0


# la distancia de un punto de la ruta a la zona prohibida
def distancia_a_pared(p):
    return dist_pared[int(limitar(round(p[0]), 0, 399)), int(limitar(round(p[1]), 0, 399))]


# ponemos una velocidad máxima para poder frenar a tiempo antes de cada curva y tramo estrecho
def velocidad_permitida(pts, llega_destino):
    n = len(pts)
    dist = 0.0     # distancia recorrida de la ruta (m)
    v_perm = min(V_MAX, V_ESPACIO_MIN + K_ESPACIO * distancia_a_pared(pts[0]))
    for i in range(1, n):
        dist += np.linalg.norm(pts[i] - pts[i - 1]) * M_CELDA
        curv = curvatura_tres_puntos(
            pts[max(0, i - 4)], pts[i], pts[min(n - 1, i + 4)]) / M_CELDA
        # el límite por la curva
        if curv < 1e-6:
            v_lim = V_MAX
        else:
            v_lim = min(V_MAX, math.sqrt(A_LAT / curv))
        # el límite por el espacio
        v_lim = min(v_lim, V_ESPACIO_MIN + K_ESPACIO * distancia_a_pared(pts[i]))
        # con tiempo de frenar
        v_perm = min(v_perm, math.sqrt(v_lim ** 2 + 2 * A_FRENO * dist))
    if llega_destino:
        v_perm = min(v_perm, math.sqrt(V_FIN ** 2 + 2 * A_FRENO * dist))
    return v_perm


# El primer punto de la ruta que está a distancia_m medidos a lo largo de la ruta
def punto_mira(pts, distancia_m):
    dist = 0.0
    for i in range(1, len(pts)):
        dist += np.linalg.norm(pts[i] - pts[i - 1]) * M_CELDA
        if dist >= distancia_m:
            return pts[i]
    return pts[-1]


mapa = WebGUI.getMap(MAP_URL)
libre = engordar_obstaculos(mapa)

# La distancia de cada celda libre a la zona prohibida y, hacia dónde crece
dist_pared = cv2.distanceTransform(libre, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
grad_f, grad_c = np.gradient(dist_pared)

# Subcampo de penalización
pen = (np.clip(D_PEN - dist_pared, 0, None) * PESO_PEN).astype(np.float32)

# Mapa de fondo (sobre el mapa original), se calcula una vez y se muestra desde el principio
img_base = imagen_del_mapa(mapa)
img = img_base    # lo que se dibuja hasta que haya destino elegido

dest_ant, total, celda_dest = None, None, None
llegado = False     # Se vuelve True cuando ya está en el destino
girando = False     # Se vuelve True mientras gira sobre sí mismo

# valores de la iteración anterior
err_ant, w_ant, v_ant = 0.0, 0.0, 0.0

while True:
    Frequency.tick()
    destino = WebGUI.getTargetPose()

    # Cuando se marca un destino nuevo, calculamos el campo
    if destino is not None and destino != dest_ant:
        dest_ant = destino
        f_dest, c_dest = mundo_a_celda(destino)
        if (not (0 <= f_dest < 400 and 0 <= c_dest < 400)
                or libre[f_dest, c_dest] != 255):
            print("Destino fuera del mapa")
        else:
            # campo = propagación + penalización
            total = propagar_desde_destino(libre, (f_dest, c_dest)) + pen
            celda_dest = (f_dest, c_dest)
            llegado = False
            girando = False
            err_ant, w_ant, v_ant = 0.0, 0.0, 0.0

            # Dibujamos la ruta completa que se sugiere desde el coche sobre el mapa de fondo
            pose0 = HAL.getPose3d()
            f_ini, c_ini = mundo_a_celda([pose0.x, pose0.y])
            img = dibujar_ruta(img_base, ruta_por_gradiente(total, f_ini, c_ini, 2000))

    # Pilotar el coche siguiendo el gradiente del campo
    if total is None or llegado:
        HAL.setV(0)         # si no hay destino o ya ha llegado = parado
        HAL.setW(0)
    else:
        # Dónde se encuentra el coche y si ha llegado al destino
        pose = HAL.getPose3d()
        f0, c0 = mundo_a_celda([pose.x, pose.y])

        if math.hypot(f0 - celda_dest[0], c0 - celda_dest[1]) < DIST_LLEGADA:
            llegado = True
            HAL.setV(0)
            HAL.setW(0)
            print("Destino alcanzado")
        else:
            # Ruta local por delante del coche
            if np.isfinite(total[f0, c0]):
                ruta_loc = ruta_por_gradiente(total, f0, c0, PASOS_LEJOS)
            else:
                # si el coche pisa la zona engordada se empieza a ir a la celda válida más cercana
                cerca = celda_alcanzable_cercana(total, f0, c0) or (f0, c0)
                ruta_loc = [(f0, c0)] + ruta_por_gradiente(total, cerca[0], cerca[1], PASOS_LEJOS)
            pts = suavizar_ruta(ruta_loc)

            # Punto de mira: más lejos cuanto más rápido va. Se elige sobre la ruta apartada de las paredes
            look = min(LOOK_MAX, LOOK_MIN + LOOK_K * v_ant)
            if girando:
                look = max(look, LOOK_GIRO)
            f1, c1 = punto_mira(alejar_de_paredes(pts), look)
            if math.hypot(f1 - f0, c1 - c0) < 0.5:   # el punto de mira coincide con el coche
                f1, c1 = celda_dest

            # Error de ángulo hacia el punto de mira (fila = x, columna = y)
            error = normalizar_angulo(math.atan2(c1 - c0, f1 - f0) - pose.yaw)

            # Vuelta sobre sí mismo
            if abs(error) > ERROR_GIRO:
                girando = True
            elif abs(error) < ERROR_FIN_GIRO:
                girando = False

            # Calculamos las velocidades lineal y angular
            if girando:
                v = 0.0
                w = math.copysign(limitar(K_GIRO * abs(error), W_GIRO_MIN, W_GIRO), error)
                err_ant, w_ant = 0.0, 0.0   # sin inercia del filtro al salir del giro
            else:
                # Velocidad angular: PD sobre el error + filtro
                # la ganancia baja por encima de V_REF
                gan = K_W * V_REF / max(v_ant, V_REF)
                w_raw = limitar(gan * error + K_D * (error - err_ant), -W_MAX, W_MAX)
                w = ALFA_W * w_raw + (1 - ALFA_W) * w_ant

                # Velocidad lineal: la que permite la curva y el espacio de la ruta
                v_perm = velocidad_permitida(pts, len(ruta_loc) - 1 < PASOS_LEJOS)
                v = min(v_perm / (1 + K_V * abs(error)), v_ant + DV_SUBE)
                err_ant, w_ant = error, w

            # Enviamos las órdenes al coche
            v_ant = v
            HAL.setV(v)
            HAL.setW(w)

    WebGUI.showNumpy(img)        # Dibujamos el mapa y su ruta
