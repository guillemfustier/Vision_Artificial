# Objetivos:
# - Rotar una imagen (parámetros centro, resize e interpolación) y trasladar

import skimage as ski
import matplotlib.pyplot as plt
from math import pi

img_original = ski.io.imread("Prácticas/practica3/images/lena256.pgm")

N_grade = int(input("Introduce el ángulo de rotación (grados): "))
N_radians = N_grade * pi / 180

# Rotación con "rotate"
img_girada = ski.transform.rotate(img_original, N_grade, center=(0, 0), resize=True, order=3)

# === Rotación con "transform" ===
# 1) Buscamos el centro de la imagen para rotarla en torno a él
# porque por defecto rotaría en torno al (0,0)
centro_y = img_original.shape[0] // 2
centro_x = img_original.shape[1] // 2

# 2) Construimos la transformación compuesta
#    - Trasladar el centro a (0,0)
#    - Rotar 90 grados (π/2 rad)
#    - Trasladar de vuelta el centro a su posición original
t_trasladar_al_origen = ski.transform.EuclideanTransform(translation=(-centro_x, -centro_y))
t_rotar_90 = ski.transform.EuclideanTransform(rotation=N_radians)
t_regresar = ski.transform.EuclideanTransform(translation=(centro_x, centro_y))

# 3) Componemos las transformaciones
transf_despl = t_trasladar_al_origen + t_rotar_90 + t_regresar

# 4) Aplicamos la transformación
img_final = ski.transform.warp(img_original, transf_despl.inverse)


fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].set_title("Original")
axs[1].imshow(img_girada, cmap=plt.cm.gray)
axs[1].set_title("Rotada (rotate)")
axs[2].imshow(img_final, cmap=plt.cm.gray)
axs[2].set_title("Rotada (transform)")


axs[0].set_axis_off()
plt.show()
