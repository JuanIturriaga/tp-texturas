import os
import cv2 as cv

def transform_dataset(image_path, output_path, size=(128,128), color_mode='grayscale', enumerate=True, square_crop=False, verbose=True):
    '''Transforma las imágenes de un dataset a un formato específico.
    Args:
        image_path (str): Ruta de la carpeta que contiene las imágenes originales.
                          Asume que el dataset está organizado en subcarpetas por clase.
                          las imágenes puede ser *.png o *.jpg
        output_path (str): Ruta de la carpeta donde se guardarán las imágenes transformadas.
        size (tuple): Tamaño al que se redimensionarán las imágenes (ancho, alto).
        color_mode (str): Modo de color para la transformación ('grayscale' o 'rgb').
        enumerate (bool): Si es True, se enumerarán las imágenes transformadas, reemplazando el nombre de archivo original.
        square_crop (bool): Si es True, recortará las imágenes para que sean cuadradas antes de redimensionarlas.
        verbose (bool): Si es True, mostrará información detallada del proceso.
    ---
    Returns:
        int: Número de imágenes transformadas.

    '''

    # Informa transformación al usuario
    if verbose:
        print(f"Transforming image: {image_path} and saving to: {output_path}")

    # Verifica que el path de salida no exista, si no existe lo crea
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise FileExistsError(f"Output path {output_path} already exists. Please choose a different path or remove the existing one.")
    
    # Contador para enumerar las imágenes transformadas
    counter = 0

    # Recorrer la carpeta de imágenes
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                counter += 1

                # Informa al usuario sobre el progreso de la transformación
                if verbose:
                    print(f"Processing image: {file} ({counter})")

                # Leer la imagen
                img = cv.imread(os.path.join(root, file))

                # Convertir a escala de grises si se especifica
                if color_mode == 'grayscale':
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Recortar la imagen para que sea cuadrada
                if square_crop:
                    height, width = img.shape[:2]
                    if height > width:
                        start = (height - width) // 2
                        img = img[start:start+width, :]
                    elif width > height:
                        start = (width - height) // 2
                        img = img[:, start:start+height]

                # Redimensionar la imagen
                img = cv.resize(img, size)

                # Crea la ruta de salida para la clase actual (subcarpeta)
                class_name = os.path.basename(root)
                class_output_path = os.path.join(output_path, class_name)
                if not os.path.exists(class_output_path):
                    os.makedirs(class_output_path)

                # Crea nombre del archivo de salida
                if enumerate:
                    output_file = os.path.join(class_output_path, f"{str(counter).zfill(5)}.png")
                else:
                    output_file = os.path.join(class_output_path, file)

                # Guardar la imagen transformada
                cv.imwrite(output_file, img)

                # Informa al usuario que la imagen ha sido guardada
                if verbose:
                    print(f"Saved transformed image to: {output_file}")

    return counter



import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def transform_dataset_to_hilbert_curve (img_dataset_path, output_path, verbose=True):
    '''Transforma las imágenes de un dataset a una curva de Hilbert y la almacena en formato numpy.
    Args:
        image_path (str): Ruta de la carpeta que contiene las imágenes originales.
                            Asume que el dataset está organizado en subcarpetas por clase.
                            las imágenes puede ser *.png o *.jpg
        output_path (str): Ruta de la carpeta donde se guardarán las imágenes transformadas.
    ---
    Returns:
        int: Número de imágenes transformadas.
    '''

    # Verifica que el path de salida no exista, si no existe lo crea
    # Si el path de salida ya existe, se informa al usuario y se omite la transformación para evitar sobrescribir datos existentes
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise FileExistsError(f"Output path {output_path} already exists. Please choose a different path or remove the existing one.")

    n = 0 # tamaño de la imagen (n x n)

    # Verifica que todas las imágenes del dataset tengan el mismo tamaño y sean cuadradas         
    for root, dirs, files in os.walk(img_dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img = cv.imread(os.path.join(root, file))
                height, width = img.shape[:2]
                if height != width:
                    raise ValueError(f"Image {file} is not square. All images must be square.")
                if n == 0:
                    n = height
                elif n != height:
                    raise ValueError(f"Image {file} has a different size than the others. All images must have the same size.")
    
    # Verifica que n sea una potencia de 2
    if (n & (n - 1)) != 0:
        raise ValueError(f"Image size {n} is not a power of 2. All images must have a size that is a power of 2.")
    
    iteration_hilbert_curve = int(np.log2(n)) # número de iteraciones para la curva de Hilbert (log2(n) para n x n)

    # Calcula la curva de Hilbert para obtener el vector 1D de la matriz
    hilbert_curve = HilbertCurve(iteration_hilbert_curve, 2)  # iteration 7 (128x128), 2 dimensiones 
    vector_hilbert = np.zeros(n*n, dtype=int)
    distances = list(range(n*n))
    points = hilbert_curve.points_from_distances(distances)

    # Recorrer la carpeta de imágenes
    counter = 0
    for root, dirs, files in os.walk(img_dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                counter += 1

                # Informa al usuario sobre el progreso de la transformación
                if verbose:
                    print(f"Processing image: {file} ({counter})")

                # Leer la imagen
                img = cv.imread(os.path.join(root, file), cv.IMREAD_GRAYSCALE)

                # Le deja solo un único canal (en caso de que la imagen tenga más de uno)
                if len(img.shape) > 2:
                    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Aplica la curva de Hilbert para obtener el vector 1D
                for i, point in enumerate(points):
                    x, y = point
                    vector_hilbert[i] = img[y, x]

                # Crea la ruta de salida para la clase actual (subcarpeta)
                class_name = os.path.basename(root)
                class_output_path = os.path.join(output_path, class_name)
                if not os.path.exists(class_output_path):
                    os.makedirs(class_output_path)

                # Crea nombre del archivo de salida
                output_file = os.path.join(class_output_path, f"{os.path.splitext(file)[0]}.npy")
                np.save(output_file, vector_hilbert)

                # Informa al usuario que la imagen ha sido guardada
                if verbose:
                    print(f"Saved transformed image to: {output_file}")

    return counter

