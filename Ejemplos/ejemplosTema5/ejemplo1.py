# Objetivos
# - fft2, ifft2, fftshift, ifftshift
# - absolute, angle, log
# - Calculo de la transformada y visualización de la magnitud y la fase

import numpy as np
import skimage as ski
import scipy.fft as fft
import matplotlib.pyplot as plt

image = ski.io.imread("images/boat.511.tiff")

# Cálculo de la transformada
ft = fft.fft2(image)

# Preparando la visualización
ftc = fft.fftshift(ft)
magnitud = np.log(np.absolute(ftc) + 1)
fase = np.angle(ftc, deg=True)

# Proceso inverso
ft_recuperada = fft.ifftshift(ftc)
img_recuperada = fft.ifft2(ft_recuperada)
rpartereal = np.real(img_recuperada)
rparteimag = np.imag(img_recuperada)

print(f"Son los resultados esperados:")
print(f"\t-> Parte real: {np.allclose(image, rpartereal)}.")
print(f"\t-> Parte imaginaria: {np.allclose(np.zeros(image.shape), rparteimag)}")


# Visualizar resultados

def ajustar0_1(img, logscale=False):
    maximo = img.max()
    minimo = img.min()
    return (img - minimo) / (maximo - minimo)


fig, axs = plt.subplots(1, 3, layout="constrained")

axs[0].imshow(image, cmap=plt.cm.gray)
axs[1].imshow(ajustar0_1(magnitud), cmap=plt.cm.gray)
axs[2].imshow(ajustar0_1(fase), cmap=plt.cm.gray)

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
