# Objetivos
# - Visualizar la funciones de base de la transformada de Fourier
# - Entender el significado de la fase dentro de la transformada

import numpy as np
import skimage as ski
import scipy.fft as fft
import matplotlib.pyplot as plt
import matplotlib


def generar_imagen_sin_fase(shape, armonicos_filas, armonicos_cols):
    ft = np.zeros(shape)
    filac = shape[0] // 2  # Asumo tamaño impar
    colc = shape[1] // 2  # Asumo tamaño impar
    ft[filac + armonicos_filas, colc + armonicos_cols] = 1
    ft[filac - armonicos_filas, colc - armonicos_cols] = 1

    ft_recuperada = fft.ifftshift(ft)
    img_recuperada = fft.ifft2(ft_recuperada)

    rpartereal = np.real(img_recuperada)
    rparteimag = np.imag(img_recuperada)
    if not np.allclose(np.zeros(ft.shape), rparteimag):
        print(f"Warning: Hay algún problema en la generación de la imagen")
    return ajustar0_1(rpartereal)


def generar_imagen_con_fase(shape, armonicos_filas, armonicos_cols, grados):
    ft = np.zeros(shape) * 1.0j
    filac = shape[0] // 2  # Asumo tamaño impar
    colc = shape[1] // 2  # Asumo tamaño impar
    ft[filac + armonicos_filas, colc + armonicos_cols] = complex(np.cos(np.radians(grados)), np.sin(np.radians(grados)))
    ft[filac - armonicos_filas, colc - armonicos_cols] = complex(np.cos(np.radians(grados)),
                                                                 -np.sin(np.radians(grados)))
    ft_recuperada = fft.ifftshift(ft)
    img_recuperada = fft.ifft2(ft_recuperada)

    rpartereal = np.real(img_recuperada)
    rparteimag = np.imag(img_recuperada)
    if not np.allclose(np.zeros(ft.shape), rparteimag):
        print(f"Warning: Hay algún problema en la genración de la imagen")
    return ajustar0_1(rpartereal)


def ajustar0_1(img):
    maximo = img.max()
    minimo = img.min()
    if np.equal(maximo, minimo):
        return img - minimo + 0.5
    else:
        return (img - minimo) / (maximo - minimo)


# Visualizar funciones de base

valores = [-5, -3, -1, 0, 1, 3, 5]
shape = (255, 255)

fig, axs = plt.subplots(len(valores), len(valores), layout="constrained")
fig.suptitle("Funciones de base")

for i in range(len(valores)):
    for j in range(len(valores)):
        imagen = generar_imagen_sin_fase(shape, valores[i], valores[j])
        axs[i, j].imshow(imagen, cmap=plt.cm.gray, norm=matplotlib.colors.Normalize(0, 1))
        axs[i, j].set_title(f"U={valores[i]} V={valores[j]}")

for ax in axs.ravel():
    ax.set_axis_off()
plt.show()

# Visualizar una función con distintas fases

U = 0
V = 2
angulos = [0, 45, 90, 135, 180, 225, 270, 315]

fig, axs = plt.subplots(2, len(angulos) // 2, layout="constrained")
fig.suptitle(f"Cambios de fase con U={U}, V={V}")
for i in range(len(angulos)):
    imagen = generar_imagen_con_fase(shape, U, V, angulos[i])
    axs[i // (len(angulos) // 2), i % (len(angulos) // 2)].imshow(imagen, cmap=plt.cm.gray)
    axs[i // (len(angulos) // 2), i % (len(angulos) // 2)].set_title(f"Fase = {angulos[i]}º")
for ax in axs.ravel():
    ax.set_axis_off()
plt.show()
