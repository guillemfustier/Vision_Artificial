# Objetivos:
# - Apreciar las diferencias entre la información captada por el sensor y la información almacenada en el archivo
# - Cálculo de errores (Maximo Error Absoluto, Error Medio, Error Cuadrático Medio)
# - Imágenes de datos ubyte (rango [0-255]) e imágenes de valores flotantes (rango [0.0-1.0])
# - Apilar matrices (np.stack)

# La imagen tablero.png fue tomada por un teléfono móvil en modo "raw".
# La imagen tablero.jpg fue la que almacenó el dispositivo para esa captura.

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np


def errorCuadráticoMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(np.power(mo2 - mo1, 2), None) / m1.size


def errorMedioBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.sum(np.abs(mo2 - mo1), None) / m1.size


def maximoErrorAbsolutoBanda(m1, m2):  # Las bandas dadas deben ser flotantes
    mo1 = m1 * 255  # Queremos medir el error en el rango [0,255]
    mo2 = m2 * 255
    return np.max(np.abs(mo2 - mo1))


imagenOriginal = ski.io.imread("images/tablero.png")
imagenGuardada = ski.io.imread("images/tablero.jpg")

imagenOriginal = ski.util.img_as_float(imagenOriginal)  # Valores flotantes en el rango [0,1]
imagenGuardada = ski.util.img_as_float(imagenGuardada)

bandas = ["Roja", "Verde", "Azul"]
for nBanda, nombre in enumerate(bandas):
    print(f"Banda {nombre}")
    print(f"   Máximo error: {maximoErrorAbsolutoBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda]):0.2f}")
    print(f"   Error medio: {errorMedioBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda]):0.2f}")
    print(f"   Error cuadrático medio: {errorCuadráticoMedioBanda(imagenOriginal[:, :, nBanda], imagenGuardada[:, :, nBanda]):0.2f}")

errores = []
for nBanda in range(3):
    errorBanda = np.abs(imagenGuardada[:, :, nBanda] - imagenOriginal[:, :, nBanda])
    errores.append(errorBanda)
errorGlobal = np.stack(errores, axis=-1)

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagenOriginal)
axs[1].imshow(imagenGuardada)
axs[2].imshow(errorGlobal)
for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
