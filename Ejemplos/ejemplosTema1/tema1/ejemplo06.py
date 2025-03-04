# Objetivos:
# - Creación de un gif animado
# - Guardar imagen en un archivo

import skimage as ski
import numpy as np
import imageio
import webbrowser

imagenOriginal = ski.io.imread("images/banderaItalia.jpg")
imagenReflejada = imagenOriginal[:, ::-1, :]

secuencia = np.stack([imagenOriginal, imagenReflejada], axis=0)

print(f"Forma de la secuencia: {secuencia.shape}")

imageio.mimsave("images/bandera.gif", secuencia, loop=0, fps=2)  # También podría ser PNG

# Abre el archivo gif desde un navegador...
webbrowser.open("images/bandera.gif")
