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
- 'embedding_method': método de incrustación utilizado ('hilbert_curve', 'raster_1', 'raster_2', 'zigzag_1', 'zigzag_2')
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
        df (pd.DataFrame): DataFrame los campos: image_file, image_id, image_size, class.
    '''

    # Informa transformación al usuario
    if verbose:
        print(f"Transforming image: {image_path} and saving to: {output_path}")

    # Verifica que el path de salida no exista, si no existe lo crea
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise FileExistsError(f"Output path {output_path} already exists. Please choose a different path or remove the existing one.")
    
    #creación de dataframe de salida
    df = pd.DataFrame(columns=['image_file', 'image_id', 'image_size', 'class'])
    
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
                df = pd.concat([df, pd.DataFrame([{'image_file': output_file, 'image_id': image_id, 'image_size': size, 'class': class_name}])], ignore_index=True)

                # Informa al usuario que la imagen ha sido guardada
                if verbose:
                    print(f"Saved transformed image to: {output_file}")

    return df



import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def embedding_images_from_dataframe (df, output_path, size=(128,128), embedding_method='hilbert_curve', verbose=True, override=False):
    '''lee las imágenes de df y devuelve otro data frame complentado archivo con el embedding y el método de embedding.
    Args:
        df (pd.DataFrame): DataFrame que contiene la información de las imágenes a transformar. Debe contener las columnas: 'image_file', 'image_id', 'image_size', 'class'.
        output_path (str): Ruta de la carpeta donde se guardarán los datos del emmbeding.
        size (tuple): Solo trabaja sobre las imágenes del df con el tamaño especificado. 
        embedding_method (str): Método de embedding a utilizar. 
            'hilbert': Aplica la curva de Hilbert para obtener el vector 1D. Requiere que las imágenes sean cuadradas y de tamaño potencia de 2 (e.g., 128x128, 256x256).
            'raster1': Aplica un recorrido raster (fila por fila) para obtener el vector 1D.
            'raster2': Aplica un recorrido raster (columna por columna) para obtener el vector 1D.
        verbose (bool): Si es True, muestra información adicional durante el proceso.
        override (bool): Si es True, permite sobrescribir el contenido del output_path si ya existe. Si es False, se lanzará un error si el output_path ya existe para evitar sobrescribir datos existentes.
    ---
    Returns:
        df_out (pd.Dataframe): Dataframe con los campos: 'image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method'
    '''

    # Verifica que el path de salida no exista, si no existe lo crea
    # Si el path de salida ya existe, se informa al usuario y se omite la transformación para evitar sobrescribir datos existentes
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        if override:
            if verbose:
                print(f"Output path {output_path} already exists. Overriding existing data.")
        else:
            raise FileExistsError(f"Output path {output_path} already exists. Please choose a different path or remove the existing one.")

    # crea un array para almacenar el embedding
    embedding = np.zeros(size[0]*size[1], dtype=int)
 
    if embedding_method == 'hilbert':
      # Verifica que el tamaño especificado sea cuadrado y potencia de 2
        if size[0] != size[1]:
            raise ValueError(f"Size {size} is not square. The embedding to Hilbert curve requires square images.")
        
        n = size[0] # tamaño de la imagen (n x n)

        if (n & (n - 1)) != 0:
            raise ValueError(f"Size {size} is not a power of 2. The embedding to Hilbert curve requires images of size that is a power of 2.")
    
        # Aplica la curva de Hilbert para obtener el vector 1D
        hilbert_curve = HilbertCurve(int(np.log2(n)), 2)  # iteration log2(n) para n x n, 2 dimensiones 
        
        distances = list(range(n*n))
        points = hilbert_curve.points_from_distances(distances)
    elif embedding_method == 'raster2':
        # Aplica un recorrido raster (columna por columna) para obtener el vector 1D
        points = [(y, x) for x in range(size[1]) for y in range(size[0])]
    else: # asume raster1
        # Aplica un recorrido raster (fila por fila) para obtener el vector 1D        
        points = [(y, x) for y in range(size[0]) for x in range(size[1])]

    #creación de dataframe de salida
    df_out = pd.DataFrame(columns=['image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method'])

    # para cada elemento del dataframe
    for index, row in df.iterrows():
        image_file = row['image_file']
        image_id = row['image_id']
        image_size = row['image_size']
        class_name = row['class']

        #convertir a tupla el tamaño de la imagen
        image_size = tuple(map(int, image_size.strip('()').split(',')))

        if image_size != size:
            if verbose:
                print(f"Skipping image: {image_file} with size {image_size} as it does not match the specified size {size}.")
            continue

        if verbose:
            print(f"Processing image: {image_file} with size {image_size} and class {class_name}")

        # Crea la ruta de salida para la clase actual (subcarpeta)
        class_output_path = os.path.join(output_path, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)

        # obtiene nombre del archivo
        image_file = image_file.replace('\\', '/') # reemplaza las barras invertidas por barras normales para evitar problemas de ruta en Windows
        image_name = image_file.split('/')[-1].split('.')[0] # obtiene el nombre del archivo sin la extensión
        
        # Crea nombre del archivo de salida
        output_file = os.path.join(class_output_path, f"{image_name}.npy")

        if verbose:
            print(f"Output file for embedding: {output_file}")

        # Leer la imagen
        img = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

        # Crea el embedding siguiendo los puntos especificados por el método de embedding
        for i, point in enumerate(points):
            x, y = point
            embedding[i] = img[y, x]

        # guarda el embedding
        np.save(output_file, embedding)

        if verbose:
            print(f"Saved embedding to: {output_file}")

        df_out = pd.concat([df_out, pd.DataFrame([{'image_file': image_file, 'image_id': image_id, 'image_size': image_size, 'class': class_name, 'embedding_file': output_file, 'embedding_method': embedding_method}])], ignore_index=True)
        
    return df_out

