import skimage as ski
import matplotlib.pyplot as plt
import math
import numpy as np

imagen = ski.io.imread("Prácticas/practica6/images/cuadros.png")

# Añadir ruido Gaussiano
gaussian_values = (0.001,
                   0.0015,
                   0.0025)
total_images = []
total_images.append(imagen)
for i in range(len(gaussian_values)):
    img_noise = ski.util.random_noise(imagen, mode="gaussian", var=gaussian_values[i])
    total_images.append(img_noise)

def canny(total_images, sigma, *args, **kwargs):
    imgs_canny = []
    for image in total_images:
        img = ski.feature.canny(image, sigma=sigma, *args, **kwargs)
        imgs_canny.append(img)
    return imgs_canny

gradientes_sobel = []
for img in total_images:
    img_grad = ski.filters.sobel(img)
    gradientes_sobel.append(img_grad)

imgs_sobel = []
imgs_hough_sobel = []
for img in gradientes_sobel:
    maximo = img.max()
    low = maximo * 0.1  # Probar otros valores
    high = maximo * 0.2
    img_borde = ski.filters.apply_hysteresis_threshold(img, low, high)
    img_sobel = ski.morphology.thin(img_borde)
    imgs_sobel.append(img_sobel)
    img_hough = ski.transform.probabilistic_hough_line(img_sobel)
    imgs_hough_sobel.append(img_hough)

imgs_canny = canny(total_images, 3)


fig, ax = plt.subplots(nrows=4, ncols=5, layout="constrained")
ax[0, 0].imshow(total_images[0], cmap='gray')
ax[1, 0].imshow(total_images[1], cmap='gray')
ax[2, 0].imshow(total_images[2], cmap='gray')
ax[3, 0].imshow(total_images[3], cmap='gray')

ax[0, 1].imshow(imgs_sobel[0], cmap='gray')
ax[1, 1].imshow(imgs_sobel[1], cmap='gray')
ax[2, 1].imshow(imgs_sobel[2], cmap='gray')
ax[3, 1].imshow(imgs_sobel[3], cmap='gray')

zeros = np.zeros(imgs_sobel[0].shape)

ax[0, 2].imshow(imgs_hough_sobel[0], cmap='gray')
ax[1, 2].imshow(imgs_hough_sobel[1], cmap='gray')
ax[2, 2].imshow(imgs_hough_sobel[2], cmap='gray')
ax[3, 2].imshow(imgs_hough_sobel[3], cmap='gray')

ax[0, 3].imshow(imgs_canny[0], cmap='gray')
ax[1, 3].imshow(imgs_canny[1], cmap='gray')
ax[2, 3].imshow(imgs_canny[2], cmap='gray')
ax[3, 3].imshow(imgs_canny[3], cmap='gray')
"""
ax[0, 4].imshow(imgs_hough_canny[0], cmap='gray')
ax[1, 4].imshow(imgs_hough_canny[1], cmap='gray')
ax[2, 4].imshow(imgs_hough_canny[2], cmap='gray')
ax[3, 4].imshow(imgs_hough_canny[3], cmap='gray')

"""
for a in ax.ravel():
    a.set_axis_off()
plt.show()