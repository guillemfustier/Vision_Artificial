import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

image = ski.io.imread("Prácticas/practica6/images/borrosa.png")

I = ski.util.img_as_float(image)
L = ski.filters.laplace(I)
alpha = 2.5
R = I + alpha*L

# Visualizar resultados
fig, axs = plt.subplots(nrows=1, ncols=2, layout="constrained")
fig.suptitle("Sharpening", fontsize=24)

axs[0].imshow(image, cmap='gray')
axs[0].set_title("Imagen original", size=16)

axs[1].imshow(R, cmap='gray')
axs[1].set_title("Realzada", size=16)

for a in axs.ravel():
    a.set_axis_off()
plt.show()