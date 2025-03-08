# Instala los paquete scikit-image y matplotlib
# Si to.do ha ido bien, el siguiente programa debe funcionar sin problemas
# De momento, no hace falta que lo entiendas...

# Objetivo: Comprobar que tienes instaladas las librerías necesarias y que to.do funciona correctamente

import skimage as ski
import matplotlib.pyplot as plt

# Asigna una imagen
image = ski.data.coins()

# Le aplica un filtro
edges = ski.filters.sobel(image)

# Muestra la imagen original y el resultado
fig, axs = plt.subplots(1, 2, layout="constrained")
axs[0].imshow(image, cmap=plt.cm.gray)
axs[1].imshow(edges, cmap=plt.cm.gray)
plt.show()
