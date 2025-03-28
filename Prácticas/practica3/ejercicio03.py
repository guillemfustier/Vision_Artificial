# Objetivos:
# - Deformación a trozos

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("Prácticas/practica3/images/Guillem.png")

src = [[0, 0],
       [0, 3544],
       [1247, 1298],  # 2- Ojo izquierdo
       [1650, 1273],  # 3- Ojo derecho
       [1444, 1500],  # 4- nariz
       [1208, 1675],  # 5- labios izqda
       [1698, 1671],  # 6- labios dcha
       [1462, 2000],  # 7- barbilla
       [345, 2258],   # 8- Hombro izquierdo
       [2574, 2365],  # 9- Hombro derecho
       [2835, 0],
       [2835, 3544]]

dst = src
src = np.array(src)
dst = np.array(dst)

dst[5] = [1143, 1603]  # Labio izquierdo
dst[6] = [1781, 1608]  # Labio derecho
dst[7] = [1462, 2150]  # Barbilla alargada
dst[8] = [345, 2108]  # Hombro izquierdo subido
dst[9] = [2574, 2215]  # Hombro derecho subido

tform = ski.transform.PiecewiseAffineTransform()
tform.estimate(src, dst)
img_t = ski.transform.warp(img_original, inverse_map=tform.inverse)

fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(img_original, cmap=plt.cm.gray)
axs[0].plot(src[:, 0], src[:, 1], '.r')
axs[1].imshow(img_t, cmap=plt.cm.gray)

axs[0].set_axis_off()
axs[1].set_axis_off()
plt.show()
