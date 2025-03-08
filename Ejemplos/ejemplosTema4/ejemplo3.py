# Objetivo:
# -Convolve (en 1D y en 2D)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

MASK_SIZE = 21
STD_DEV = 3

imagen = ski.io.imread("images/boat.512.tiff")
imagen = ski.util.img_as_float(imagen)

vector = scipy.signal.windows.gaussian(MASK_SIZE, STD_DEV)
vector /= np.sum(vector)

vectorH = vector.reshape(1, MASK_SIZE)
vectorV = vector.reshape(MASK_SIZE, 1)

matriz = vectorV @ vectorH

inicio2D = time.time()
res_convol2D = scipy.ndimage.convolve(imagen, matriz)
fin2D = time.time()

inicio1D = time.time()
resH = scipy.ndimage.convolve(imagen, vectorH)
res1D = scipy.ndimage.convolve(resH, vectorV)
fin1D = time.time()

tiempo2D = fin2D - inicio2D
tiempo1D = fin1D - inicio1D

print("¿Obtenemos el mismo resultado?", np.allclose(res_convol2D, res1D))
print(f"Tiempo empleado con máscara  2D: {tiempo2D:0.9f}")
print(f"Tiempo empleado con máscaras 1D: {tiempo1D:0.9f}")
print(f"Factor 2D/1D: {tiempo2D / tiempo1D:0.2f}")

fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(imagen, cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(res_convol2D, cmap='gray')
axs[1].set_title("Convolución 2D")
axs[2].imshow(res1D, cmap='gray')
axs[2].set_title("Convoluciones 1D")
for a in axs:
    a.set_axis_off()
plt.show()
