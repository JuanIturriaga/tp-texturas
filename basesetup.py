import os
import cv2 as cv
import numpy as np
import pandas as pd

"""
Formato del dataframe con el que se va a trabajar 

columnas: 
- 'image_file': ruta y nombre del archivo la imagen procesada.
- 'image_id': identificador único de la imagen.
- 'image_size': tamaño de la imagen procesada (ancho x alto).
- 'class': clase a la que pertenece la imagen. (aluminium_foil, brown_bread, corduroy, etc...)
- 'embedding_file': ruta y nombre del archivo de la incrustación (embedding) de la imagen.
- 'embedding_method': método de incrustación utilizado ('hilbert_curve', 'raster-1', 'raster-2', 'zigzag-1', 'zigzag-2')
- 'embedding_dimension' (m): dimensión de los patrones ordinales.
- 'time_delay' (tau): retardo temporal utilizado para calcular los patrones ordinales.
- 'ordinal_pattern': patrón ordinal representado como una tupla de enteros.
- 'entropy_type': tipo de entropía calculada ('shannon', 'renyi', 'tsallis').
- 'entropy_parameter' (alpha o q): parámetro utilizado para calcular la entropía de 'renyi' o 'tsallis'.
- 'entropy_value': valor numérico de la entropía normalizada calculada.
- 'complexity_value': valor numérico de la complejidad calculada.

"""

def transform_dataset(image_path, output_path, size=(128,128), color_mode='grayscale', enumerate=True, square_crop=True, verbose=True):
    '''Transforma las imágenes de un dataset a un formato específico y devuelve un dataframe con la información de las imágenes transformadas.
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
        df (pd.DataFrame): DataFrame (con todos los campos) con la información de las imágenes transformadas: image_file, image_id, image_size, class.
    '''

    # Informa transformación al usuario
    if verbose:
        print(f"Transforming image: {image_path} and saving to: {output_path}")

    # Verifica que el path de salida no exista, si no existe lo crea
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise FileExistsError(f"Output path {output_path} already exists. Please choose a different path or remove the existing one.")
    
    #creación de dataframe
    df = pd.DataFrame(columns=['image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method', 'embedding_dimension', 'time_delay', 'ordinal_pattern', 'entropy_type', 'entropy_parameter', 'entropy_value', 'complexity_value'])

    
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

                # Redimensionar la imagen si es requerido y obtener el tamaño final de la imagen transformada
                if size is not None:
                    img = cv.resize(img, size)
                else:
                    size = img.shape[:2][::-1]  # (ancho, alto)

                # Crea la ruta de salida para la clase actual (subcarpeta)
                class_name = os.path.basename(root)
                class_output_path = os.path.join(output_path, class_name)
                if not os.path.exists(class_output_path):
                    os.makedirs(class_output_path)

                # Crea nombre del archivo de salida
                if enumerate:
                    output_file = os.path.join(class_output_path, f"{str(counter).zfill(5)}.png")
                    image_id = str(counter).zfill(5)
                else:
                    output_file = os.path.join(class_output_path, file)
                    image_id = os.path.splitext(file)[0]

                # Guardar la imagen transformada
                cv.imwrite(output_file, img)

                # Agregar al df los datos
                df = df.append({'image_file': output_file, 'image_id': image_id, 'image_size': size, 'class': class_name}, ignore_index=True)

                # Informa al usuario que la imagen ha sido guardada
                if verbose:
                    print(f"Saved transformed image to: {output_file}")

    return counter



import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def embedding_hilbert_curve (df, output_path, size=(128,128), mode='fill', verbose=True):
    '''rellena los campos del dataframe correspondientes al embeddig y la almacena los datos formato numpy.
    Args:
        df (pd.DataFrame): DataFrame que contiene la información de las imágenes a transformar.
        output_path (str): Ruta de la carpeta donde se guardarán los datos del emmbeding de la curva de Hilbert.
        size (tuple): Solo trabaja sobre las imágenes del df con el tamaño especificado. Tamaño de las imágenes (deben ser cuadradas y de tamaño potencia de 2).
        mode (str): Modo de trabajo: 
            - 'fill': rellena los campos del dataframe correspondientes al embedding y guarda los datos en formato numpy.
            - 'append': toma los datos de todas las imágenes con un embedding diferente a 'hilbert_curve' y los agregar al dataframe con los campos correspondientes al embedding de la curva de Hilbert.
            - 'overwrite': sobrescribe los campos del dataframe correspondientes al embedding de la curva de Hilbert con los datos del embedding de la curva de Hilbert.
        verbose (bool): Si es True, muestra información adicional durante el proceso.
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

    # Verifica que el tamaño especificado sea cuadrado y potencia de 2
    if size[0] != size[1]:
        raise ValueError(f"Size {size} is not square. The embedding to Hilbert curve requires square images.")

    n = size[0] # tamaño de la imagen (n x n)

    if (n & (n - 1)) != 0:
        raise ValueError(f"Size {size} is not a power of 2. The embedding to Hilbert curve requires images of size that is a power of 2.")
    
    # Aplica la curva de Hilbert para obtener el vector 1D
    hilbert_curve = HilbertCurve(int(np.log2(n)), 2)  # iteration log2(n) para n x n, 2 dimensiones 
    embedding = np.zeros(n*n, dtype=int)
    distances = list(range(n*n))
    points = hilbert_curve.points_from_distances(distances)

    # para cada elemento del dataframe
    for index, row in df.iterrows():
        image_file = row['image_file']
        image_id = row['image_id']
        image_size = row['image_size']
        class_name = row['class']
        if verbose:
            print(f"Processing image: {image_file} with size {image_size} and class {class_name}")

        if (mode == 'fill'):
            if row['embedding_method'] == '':
                row['embedding_method'] = 'hilbert_curve'
        
                #continuar
        
        #todo: mode 'append' y 'overwrite'

        

        # Verifica que la imagen tenga el tamaño especificado
        if image_size != size:
            if verbose:
                print(f"Skipping image {image_file} with size {image_size} (expected size: {size})")
            continue

        # Leer la imagen
        img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

        # Aplica la curva de Hilbert para obtener el vector 1D
        hilbert_curve = HilbertCurve(int(np.log2(n)), 2)  # iteration log2(n) para n x n, 2 dimensiones 
        vector_hilbert = np.zeros(n*n, dtype=int)
        distances = list(range(n*n))
        points = hilbert_curve.points_from_distances(distances)

        for i, point in enumerate(points):
            x, y = point
            vector_hilbert[i] = img[y, x]

        # Crea la ruta de salida para la clase actual (subcarpeta)
        class_output_path = os.path.join(output_path, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        # Crea nombre del archivo de salida
        output_file = os.path.join(class_output_path, f"{image_id}.npy")
        np.save(output_file, vector_hilbert)

        # Actualiza el dataframe con los campos correspondientes al embedding de la curva de Hilbert
        df.at[index, 'embedding_file'] = output_file
        df.at[index, 'embedding_method'] = 'hilbert_curve'
        df.at[index, 'embedding_dimension'] = n*n

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

