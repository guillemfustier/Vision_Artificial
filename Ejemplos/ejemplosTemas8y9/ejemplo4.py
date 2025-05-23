import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

cruz = np.zeros((256, 256))
rectangulo_shape = (32, 192)
fila_ini = (cruz.shape[0] - rectangulo_shape[0]) // 2
col_ini = (cruz.shape[1] - rectangulo_shape[1]) // 2
cruz[fila_ini:fila_ini + rectangulo_shape[0], col_ini:col_ini + rectangulo_shape[1]] = 1
cruz[col_ini:col_ini + rectangulo_shape[1], fila_ini:fila_ini + rectangulo_shape[0]] = 1

caballo = ski.data.horse()
caballo = ski.util.invert(caballo)

esqueleto_cruz = ski.morphology.skeletonize(cruz, method="lee")
esqueleto_caballo = ski.morphology.skeletonize(caballo, method="lee")

convex_cruz = ski.morphology.convex_hull_image(cruz)
convex_caballo = ski.morphology.convex_hull_image(caballo)

# Visualizar resultados
fig, axs = plt.subplots(2, 3, layout="constrained")
for ax in axs.ravel():
    ax.set_axis_off()
fig.suptitle("Skeleton & Convex hull", fontsize=24)

axs[0, 0].imshow(cruz, cmap="gray")
axs[0, 0].set_title("Rectángulo", fontsize=16)

axs[0, 1].imshow(esqueleto_cruz, cmap="gray")
axs[0, 1].set_title("Skeleton", fontsize=16)

axs[0, 2].imshow(convex_cruz, cmap="gray")
axs[0, 2].set_title("Convex hull", fontsize=16)

axs[1, 0].imshow(caballo, cmap="gray")
axs[1, 0].set_title("Caballo", fontsize=16)

axs[1, 1].imshow(esqueleto_caballo, cmap="gray")
axs[1, 1].set_title("Skeleton", fontsize=16)

axs[1, 2].imshow(convex_caballo, cmap="gray")
axs[1, 2].set_title("Convex hull", fontsize=16)

plt.show()
