import WebGUI
import HAL
import Frequency
import numpy as np
import cv2
import heapq
import math

# ---------------- PARÁMETROS ----------------
MAP_URL = '/resources/exercises/global_navigation/images/cityLargenBin.png'
VALOR_CARRETERA = 255
M_POR_CELDA = 1.25     # metros por celda (500 m / 400 celdas)

# Campo: engordado de obstáculos y penalización por cercanía
N_CELDAS = 4           # celdas que se engordan los obstáculos
R_PEN = 12             # radio de la penalización (celdas)
PESO_PEN = 0.7         # pendiente de la penalización (< 1)

# Ruta local por delante del coche
PASOS_LEJOS = 36       # horizonte (celdas)
D_SEG = 5.0            # distancia a la pared hasta la que se aparta la ruta (celdas)
PUSH_PASOS = 3         # máximo que se aparta cada punto (celdas)

# Velocidad lineal
V_MAX = 25.0
A_LAT = 6.5            # aceleración lateral admisible (sube = curvas más rápidas)
A_FRENO = 6.0          # deceleración supuesta (baja = frena antes)
V_FIN = 2.0            # velocidad al llegar al destino
K_V = 0.8              # frenada extra según el error de ángulo
DV_SUBE = 0.3          # aumento máximo de V por iteración
V_ESPACIO_MIN = 5.0    # velocidad máxima pegado a la pared ESTABA EN 3.0
K_ESPACIO = 4.0        # m/s extra por cada celda de holgura

# Punto de mira (metros): LOOK_MIN + LOOK_K * V, con tope LOOK_MAX
LOOK_MIN, LOOK_MAX, LOOK_K = 6.0, 20.0, 0.7
LOOK_GIRO = 12.0       # mira mientras gira sobre sí mismo

# Velocidad angular
K_W = 1.0
V_REF = 15.0           # por encima de esta V la ganancia de W baja
K_D = 0.4
ALFA_W = 0.5           # filtro de W (1 = sin filtro)
W_MAX = 2.5

# Vuelta en U (girar sobre sí mismo)
ERROR_GIRO = 1.2       # error (rad) a partir del cual gira en el sitio
ERROR_FIN_GIRO = 0.15  # error (rad) por debajo del cual vuelve a avanzar
W_GIRO, W_GIRO_MIN, K_GIRO = 2.0, 0.4, 2.0

DIST_LLEGADA = 3       # celdas al destino para darlo por alcanzado

# (df, dc, longitud del paso); los rectos primero: en un empate gana el recto
VECINOS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]


# ---------------- UTILIDADES ----------------
def limitar(x, minimo, maximo):
    return max(minimo, min(maximo, x))


def normalizar(a):
    """Ángulo en (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def mundo_a_celda(punto):
    """Mundo (x, y en metros) -> (fila, col)."""
    col, fila = WebGUI.worldToGrid(punto)
    return fila, col


# ---------------- PLANIFICACIÓN ----------------
def engordar(mapa):
    """Agranda los obstáculos N_CELDAS: zona libre = 255."""
    obstaculos = (mapa != VALOR_CARRETERA).astype(np.uint8)
    tam = 2 * N_CELDAS + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam))
    return np.where(cv2.dilate(obstaculos, kernel) == 0, 255, 0).astype(np.uint8)


def propagar(libre, destino):
    """Subcampo de propagación: distancia al destino (Dijkstra, 8 vecinos)."""
    filas, cols = libre.shape
    ancho = cols + 2                                    # borde de 1 celda prohibida
    libre_l = np.pad(libre == 255, 1, constant_values=False).ravel().tolist()
    vecinos = [(df * ancho + dc, largo) for df, dc, largo in VECINOS]

    campo = [float('inf')] * len(libre_l)
    inicio = (destino[0] + 1) * ancho + (destino[1] + 1)
    campo[inicio] = 0.0
    cola = [(0.0, inicio)]
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


def ruta_gradiente(total, f, c, max_pasos):
    """Camino bajando por el campo. En cada paso elige el vecino con mayor bajada
    por unidad de longitud (el que mejor sigue el gradiente)."""
    filas, cols = total.shape
    ruta = [(f, c)]
    for _ in range(max_pasos):
        mejor, mejor_pend = None, 0.0
        for df, dc, largo in VECINOS:
            nf, nc = f + df, c + dc
            if 0 <= nf < filas and 0 <= nc < cols and total[nf, nc] < total[f, c]:
                pend = (total[f, c] - total[nf, nc]) / largo
                if pend > mejor_pend + 1e-9:
                    mejor, mejor_pend = (nf, nc), pend
        if mejor is None:                               # mínimo alcanzado: el destino
            break
        f, c = mejor
        ruta.append(mejor)
    return ruta


def celda_alcanzable_cercana(total, f, c, radio=8):
    """Celda con campo finito más próxima (por si el coche pisa la zona engordada)."""
    f0, c0 = max(0, f - radio), max(0, c - radio)
    filas, cols = np.nonzero(np.isfinite(total[f0:f + radio + 1, c0:c + radio + 1]))
    if len(filas) == 0:
        return None
    i = np.argmin((filas + f0 - f) ** 2 + (cols + c0 - c) ** 2)
    return int(filas[i] + f0), int(cols[i] + c0)


def campo_a_imagen(campo, ruta):
    """Campo (más claro cerca del destino) con la ruta en blanco."""
    img = np.zeros(campo.shape, dtype=np.uint8)
    validas = np.isfinite(campo)
    img[validas] = 40 + 160 * (1 - campo[validas] / campo[validas].max())
    for rf, rc in ruta:
        img[max(0, rf - 1):rf + 2, max(0, rc - 1):rc + 2] = 255
    return img


# ---------------- PILOTAJE ----------------
def suavizar(ruta, ventana=7):
    """Media móvil sobre la ruta: quita el zigzag de la rejilla."""
    pts = np.array(ruta, dtype=float)
    m = ventana // 2
    return np.array([pts[max(0, i - m):i + m + 1].mean(axis=0) for i in range(len(pts))])


def alejar_de_paredes(pts):
    """Aparta cada punto de la ruta de las paredes (hacia donde crece la distancia a
    ellas) hasta D_SEG, el eje de la calle o PUSH_PASOS. Sirve para elegir el punto
    de mira: el coche abre la curva antes de la esquina."""
    filas, cols = dist_pared.shape
    salida = pts.copy()
    for k in range(len(salida)):
        p = salida[k].copy()
        for _ in range(PUSH_PASOS):
            r, q = int(round(p[0])), int(round(p[1]))
            if not (0 <= r < filas and 0 <= q < cols) or dist_pared[r, q] >= D_SEG:
                break
            g = np.array([grad_f[r, q], grad_c[r, q]])
            if np.linalg.norm(g) < 0.3:                 # eje de la calle
                break
            nuevo = p + g / np.linalg.norm(g)           # un paso de 1 celda
            rn, qn = int(round(nuevo[0])), int(round(nuevo[1]))
            if not (0 <= rn < filas and 0 <= qn < cols) or dist_pared[rn, qn] <= dist_pared[r, q]:
                break
            p = nuevo
        salida[k] = p
    return salida


def curvatura(a, b, c):
    """Curvatura (1/celdas) de la circunferencia que pasa por tres puntos."""
    cruz = abs((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0])
    den = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
    return 2.0 * cruz / den if den > 1e-9 else 0.0


def holgura(p):
    """Distancia (celdas) de un punto de la ruta a la zona prohibida."""
    return dist_pared[int(limitar(round(p[0]), 0, 399)), int(limitar(round(p[1]), 0, 399))]


def velocidad_permitida(pts, llega_destino):
    """Velocidad máxima ahora para poder frenar a tiempo antes de cada curva y cada
    tramo estrecho de la ruta local (y antes del destino si está a la vista)."""
    n = len(pts)
    s = 0.0                                             # distancia a lo largo de la ruta (m)
    v_perm = min(V_MAX, V_ESPACIO_MIN + K_ESPACIO * holgura(pts[0]))
    for i in range(1, n):
        s += np.linalg.norm(pts[i] - pts[i - 1]) * M_POR_CELDA
        k = curvatura(pts[max(0, i - 4)], pts[i], pts[min(n - 1, i + 4)]) / M_POR_CELDA
        v_i = V_MAX if k < 1e-6 else min(V_MAX, math.sqrt(A_LAT / k))
        v_i = min(v_i, V_ESPACIO_MIN + K_ESPACIO * holgura(pts[i]))
        v_perm = min(v_perm, math.sqrt(v_i ** 2 + 2 * A_FRENO * s))
    if llega_destino:
        v_perm = min(v_perm, math.sqrt(V_FIN ** 2 + 2 * A_FRENO * s))
    return v_perm


def punto_mira(pts, dist_m):
    """Primer punto de la ruta a dist_m metros a lo largo de ella."""
    s = 0.0
    for i in range(1, len(pts)):
        s += np.linalg.norm(pts[i] - pts[i - 1]) * M_POR_CELDA
        if s >= dist_m:
            return pts[i]
    return pts[-1]


# ---------------- CÓDIGO SECUENCIAL ----------------
mapa = WebGUI.getMap(MAP_URL)
libre = engordar(mapa)
dist_pared = cv2.distanceTransform(libre, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
grad_f, grad_c = np.gradient(dist_pared)                # hacia dónde crece la distancia a la pared
pen = (np.clip(R_PEN - dist_pared, 0, None) * PESO_PEN).astype(np.float32)   # subcampo de penalización
imagen = libre

destino_anterior, total, celda_destino = None, None, None
llegado, girando = False, False
error_prev, W_prev, V_prev = 0.0, 0.0, 0.0

# ---------------- BUCLE REACTIVO ----------------
while True:
    Frequency.tick()
    destino = WebGUI.getTargetPose()

    # Planificación: solo cuando el usuario marca un destino nuevo
    if destino is not None and destino != destino_anterior:
        destino_anterior = destino
        fila, col = mundo_a_celda(destino)
        if not (0 <= fila < 400 and 0 <= col < 400) or libre[fila, col] != 255:
            print("Destino fuera del mapa o de la carretera")
        else:
            total = propagar(libre, (fila, col)) + pen  # campo = propagación + penalización
            celda_destino = (fila, col)
            llegado, girando = False, False
            error_prev, W_prev, V_prev = 0.0, 0.0, 0.0
            p = HAL.getPose3d()
            imagen = campo_a_imagen(total, ruta_gradiente(total, *mundo_a_celda([p.x, p.y]), 2000))

    # Pilotaje: seguir el gradiente del campo
    if total is None or llegado:
        HAL.setV(0)
        HAL.setW(0)
    else:
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
            else:       # coche en la zona engordada: ir a la celda válida más cercana
                cerca = celda_alcanzable_cercana(total, f0, c0) or (f0, c0)
                ruta_loc = [(f0, c0)] + ruta_gradiente(total, cerca[0], cerca[1], PASOS_LEJOS)
            pts = suavizar(ruta_loc)

            # Punto de mira: más lejos cuanto más rápido va; sobre la ruta apartada de las paredes
            look = min(LOOK_MAX, LOOK_MIN + LOOK_K * V_prev)
            if girando:
                look = max(look, LOOK_GIRO)
            f1, c1 = punto_mira(alejar_de_paredes(pts), look)
            if math.hypot(f1 - f0, c1 - c0) < 0.5:      # mira == coche: ir al destino
                f1, c1 = celda_destino

            # Error de ángulo hacia el punto de mira (fila ~ x, col ~ y)
            error = normalizar(math.atan2(c1 - c0, f1 - f0) - pose.yaw)

            # Histéresis de la vuelta en U
            if abs(error) > ERROR_GIRO:
                girando = True
            elif abs(error) < ERROR_FIN_GIRO:
                girando = False

            if girando:     # parado, girando sobre sí mismo
                V = 0.0
                W = math.copysign(limitar(K_GIRO * abs(error), W_GIRO_MIN, W_GIRO), error)
                error_prev, W_prev = 0.0, 0.0
            else:
                # W: PD sobre el error + filtro; la ganancia baja por encima de V_REF
                k_w = K_W * V_REF / max(V_prev, V_REF)
                W_raw = limitar(k_w * error + K_D * (error - error_prev), -W_MAX, W_MAX)
                W = ALFA_W * W_raw + (1 - ALFA_W) * W_prev
                # V: la que permiten curvatura y holgura de la ruta, menos frenada por el error
                v_perm = velocidad_permitida(pts, len(ruta_loc) - 1 < PASOS_LEJOS)
                V = min(v_perm / (1 + K_V * abs(error)), V_prev + DV_SUBE)
                error_prev, W_prev = error, W

            V_prev = V
            HAL.setV(V)
            HAL.setW(W)

    WebGUI.showNumpy(imagen)
