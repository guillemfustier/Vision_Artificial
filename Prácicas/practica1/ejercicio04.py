# Objetivos:
# - Creación de un gif animado
# - Guardar imagen en un archivo

import skimage as ski
import numpy as np
import imageio
import webbrowser

imagen1 = ski.io.imread("images/flecha_transparente.png")
imagen1[:, :, 3] = 255
imagen4 = np.rot90(imagen1, k=1)
imagen3 = np.rot90(imagen4, k=1)
imagen2 = np.rot90(imagen3, k=1)

secuencia = np.stack([imagen1, imagen2, imagen3, imagen4], axis=0)

print(f"Forma de la secuencia: {secuencia.shape}")

imageio.mimsave("images/flecha.gif", secuencia, loop=0, fps=2)  # También podría ser PNG

# Abre el archivo gif desde un navegador...
webbrowser.open("images/flecha.gif")
