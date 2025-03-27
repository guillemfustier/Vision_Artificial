# Objetivos:
# - Crear, manejar y aplicar transformaciones afines (también podrían ser euclídeas o de similitud)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("Prácticas/practica3/images/lena256.pgm")

FACTOR_ESCALADO = (0.5, 0.75)
ta = ski.transform.AffineTransform(scale=FACTOR_ESCALADO)
mscalado = ta.params
print(f"Matriz de escalado con factor {FACTOR_ESCALADO}")
print(mscalado)
print()

DISTANCIA_TRASLACION = (64, 0)
ta = ski.transform.AffineTransform(translation=DISTANCIA_TRASLACION)
mtranslacion = ta.params
print(f"Matriz de traslación con distancia {DISTANCIA_TRASLACION}")
print(mtranslacion)
print()

ANGULO_ROTACION_EN_GRADOS = 15
ta = ski.transform.AffineTransform(rotation=np.radians(ANGULO_ROTACION_EN_GRADOS))
mrotacion = ta.params
print(f"Matriz de rotación con ángulo {ANGULO_ROTACION_EN_GRADOS}")
print(mrotacion)
print()

ANGULO_INCLINACION_EN_GRADOS = -10
ta = ski.transform.AffineTransform(shear=np.radians(ANGULO_INCLINACION_EN_GRADOS))
mshear = ta.params
print(f"Matriz de inclinación con ángulo de {ANGULO_INCLINACION_EN_GRADOS} grados")
print(mshear)
print()

matriz_total_calculada = mshear @ mrotacion @ mtranslacion @ mscalado

# Convertir la transformación afín en proyectiva
# matriz_total_calculada[2, 0] = 0.0025
# matriz_total_calculada[2, 1] = -0.0025

print(f"Matriz de transformación total calculada: 1. Escalado 2. Inclinacion 3. Rotación 4. Traslación")
print(matriz_total_calculada)
print()

mita = ski.transform.AffineTransform(matriz_total_calculada)
mi_img_tras = ski.transform.warp(img_original, mita.inverse)

ta = ski.transform.AffineTransform(scale=FACTOR_ESCALADO, translation=DISTANCIA_TRASLACION,rotation=np.radians(ANGULO_ROTACION_EN_GRADOS),
                                   shear=np.radians(ANGULO_INCLINACION_EN_GRADOS))
print(f"Matriz de transformación generada con todos los paŕametros")
print(ta.params)
print()

img_tras = ski.transform.warp(img_original, ta.inverse)

fig, axs = plt.subplots(1, 3, layout="constrained")

axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].set_title("Original")
axs[1].imshow(mi_img_tras, cmap=plt.cm.gray)
axs[1].set_title("Imagen calculada (TOTAL)")
axs[2].imshow(img_tras, cmap=plt.cm.gray)
axs[2].set_title("Matriz generada (TOTAL)")

axs[0].set_axis_off()
plt.show()

""" ¿POR QUÉ NO SON EXACTAMENTE IGUALES? En teoría debería salir un poco más difuminada
La matriz generada está un poco desplazada hacia arriba en comparación con la matriz 
calculada. Esto se debe 

"""