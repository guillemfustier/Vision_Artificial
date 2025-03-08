# Objetivos:
# - Deformación a trozos

import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

img_original = ski.io.imread("images/lena.ppm")

src = [[0, 0],
       [0, 511],
       [264, 264],  # Ojo izquierdo
       [326, 264],  # Ojo derecho
       [314, 320],  # nariz
       [336, 328],  # moflete
       [266, 350],  # labios izqda
       [318, 348],  # labios dcha
       [290, 382],  # barbilla
       [511, 0],
       [511, 511]]

dst = src
src = np.array(src)
dst = np.array(dst)

dst[4] = [324, 330]  # Nariz alargada

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
