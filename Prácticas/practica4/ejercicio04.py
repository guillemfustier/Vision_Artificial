# Objetivo:
# -Convolve (en 1D y en 2D)

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time

MASK_SIZE = 21
STD_DEV = 3
alpha = 1.2

# Imagen Original I
I = ski.io.imread("Prácticas/practica4/images/borrosa.png")
I = ski.util.img_as_float(I)

# Máscara de convolución del filtro gaussiano
vector = scipy.signal.windows.gaussian(MASK_SIZE, STD_DEV)
vector /= np.sum(vector)
vectorH = vector.reshape(1, MASK_SIZE)
vectorV = vector.reshape(MASK_SIZE, 1)
matriz = vectorV @ vectorH

# Convolución 2D -> Imagen filtrada F
resH = scipy.ndimage.convolve(I, vectorH)
F = scipy.ndimage.convolve(resH, vectorV)

# Imagen final R
R = I + alpha * (I - F)

# Mostrar resultados
fig, axs = plt.subplots(1, 3, layout="constrained")
axs[0].imshow(I, cmap='gray')
axs[0].set_title("Original (I)")
axs[1].imshow(F, cmap='gray')
axs[1].set_title("Filtrada (F)")
axs[2].imshow(R, cmap='gray')
axs[2].set_title("Final (R)")
for a in axs:
    a.set_axis_off()
plt.show()