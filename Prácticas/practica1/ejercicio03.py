# Objetivos:
# - Almacenar archivos JPG con diferentes calidades.
# - Analizar los errores cometidos en cada caso y el tamaño del archivo resultante

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os


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


calidades = [15, 75, 100];  # Porcentaje de calidad al guardar el JPG

imagenOriginalPNG = ski.io.imread("images/mapas.png")
for calidad in calidades:
    imageio.imsave(f"images/mapas{calidad}.jpg", imagenOriginalPNG, quality=calidad)
    imagenGuardadaJPG = ski.io.imread(f"images/mapas{calidad}.jpg")

    imagenOriginal = ski.util.img_as_float(imagenOriginalPNG)  # Valores flotantes en el rango [0,1]
    imagenGuardada = ski.util.img_as_float(imagenGuardadaJPG)

    print(f"\nCALIDAD: {calidad}%  Tamaño PNG: {os.path.getsize('images/mapas.png')} Tamaño JPG: {os.path.getsize(f'images/mapas{calidad}.jpg')}")
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
