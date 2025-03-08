# Objetivos:
# - Crear, manejar y aplicar transformaciones afines (también podrían ser euclídeas o de similitud)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena256.pgm")

FACTOR_ESCALADO = 1.5
ta = ski.transform.AffineTransform(scale=FACTOR_ESCALADO)
mscalado = ta.params
print(f"Matriz de escalado con factor {FACTOR_ESCALADO}")
print(mscalado)
print()

ANGULO_INCLINACION_EN_GRADOS = 10
ta = ski.transform.AffineTransform(shear=np.radians(ANGULO_INCLINACION_EN_GRADOS))
mshear = ta.params
print(f"Matriz de inclinación con ángulo de {ANGULO_INCLINACION_EN_GRADOS} grados")
print(mshear)
print()

ANGULO_ROTACION_EN_GRADOS = 45
ta = ski.transform.AffineTransform(rotation=np.radians(ANGULO_ROTACION_EN_GRADOS))
mrotacion = ta.params
print(f"Matriz de rotación con ángulo {ANGULO_ROTACION_EN_GRADOS}")
print(mrotacion)
print()

DISTANCIA_TRASLACION = (256, 0)
ta = ski.transform.AffineTransform(translation=DISTANCIA_TRASLACION)
mtranslacion = ta.params
print(f"Matriz de traslación con distancia {DISTANCIA_TRASLACION}")
print(mtranslacion)
print()

matriz_total_calculada = mtranslacion @ mrotacion @ mshear @ mscalado

# Convertir la transformación afín en proyectiva
# matriz_total_calculada[2, 0] = 0.0025
# matriz_total_calculada[2, 1] = -0.0025

print(f"Matriz de transformación total calculada: 1. Escalado 2. Inclinacion 3. Rotación 4. Traslación")
print(matriz_total_calculada)
print()

mita = ski.transform.AffineTransform(matriz_total_calculada)
mi_img_tras = ski.transform.warp(img_original, mita.inverse, output_shape=(512, 512))

ta = ski.transform.AffineTransform(scale=FACTOR_ESCALADO, shear=np.radians(ANGULO_INCLINACION_EN_GRADOS),
                                   rotation=np.radians(ANGULO_ROTACION_EN_GRADOS), translation=DISTANCIA_TRASLACION)
print(f"Matriz de transformación generada con todos los paŕametros")
print(ta.params)
print()

img_tras = ski.transform.warp(img_original, ta.inverse, output_shape=(512, 512))

fig, axs = plt.subplots(1, 3, layout="constrained")

axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].set_title("Original")
axs[1].imshow(mi_img_tras, cmap=plt.cm.gray)
axs[1].set_title("Matriz calculada")
axs[2].imshow(img_tras, cmap=plt.cm.gray)
axs[2].set_title("Matriz generada")

axs[0].set_axis_off()
plt.show()
