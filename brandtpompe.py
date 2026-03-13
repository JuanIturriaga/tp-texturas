import ordpy.ordpy as ordpy
import numpy as np
import os
import pandas as pd

"""
Formato del dataframe con el que se va a trabajar 

columnas: 
- 'image_file': ruta y nombre del archivo la imagen procesada.
- 'image_id': identificador único de la imagen.
- 'image_size': tamaño de la imagen procesada (ancho x alto).
- 'class': clase a la que pertenece la imagen. (aluminium_foil, brown_bread, corduroy, etc...)
- 'embedding_file': ruta y nombre del archivo de la incrustación (embedding) de la imagen.
- 'embedding_method': método de incrustación utilizado (Hilbert Curve, raster-1, raster-2, zigzag-1, zigzag-2)
- 'embedding_dimension' (m): dimensión de los patrones ordinales.
- 'time_delay' (tau): retardo temporal utilizado para calcular los patrones ordinales.
- 'ordinal_pattern': patrón ordinal representado como una tupla de enteros.
- 'ordinal_probability': frecuencia del patrón ordinal.
- 'entropy_type': tipo de entropía calculada (Shannon, Renyi, Tsallis).
- 'entropy_parameter' (alpha o q): parámetro utilizado para calcular la entropía de Renyi o Tsallis.
- 'entropy_value': valor numérico de la entropía normalizada calculada.
- 'complexity_value': valor numérico de la complejidad calculada.

"""


def ordinal_distribution_save (df, m=3, tau=1, verbose=False):
    """
    Crea y devuelve un DataFrame con la distribución ordinal.

    Args:
        df (pd.DataFrame): DataFrame que debe contener los embeddings de las imágenes y sus respectivas clases.
            Debe contener los campos 'image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method'
        m (int): Orden de los patrones ordinales.
        tau (int): Retardo temporal.
        verbose (bool): Si es True, muestra información adicional durante el proceso.
    ---
    Returns:
        pd.DataFrame: DataFrame con la distribución ordinal.
            Con los mismos campos de entrada y agrega: 'embedding_dimension', 'time_delay', 'ordinal_pattern', 'ordinal_probability'
    """
    import pandas as pd

    result = []

    # Itera sobre cada fila del DataFrame de entrada
    for index, row in df.iterrows():
        image_file = row['image_file']
        image_id = row['image_id']
        image_size = row['image_size']
        image_class = row['class']
        embedding_file = row['embedding_file']
        embedding_method = row['embedding_method']

        # Carga el embedding de la imagen
        embedding = np.load(embedding_file)

        # Calcula la distribución ordinal utilizando ordpy
        ordinal_patterns, ordinal_probability = ordpy.ordinal_distribution(embedding, dx=m, taux=tau, return_missing=True)
        
        string_pattern = ''
        for pattern in ordinal_patterns:
            string_pattern += ''.join(map(str, pattern)) + ' '

        # Agrega los resultados al DataFrame de salida        
        result.append({
            'image_file': image_file,
            'image_id': image_id,
            'image_size': image_size,
            'class': image_class,
            'embedding_file': embedding_file,
            'embedding_method': embedding_method,
            'embedding_dimension': m,
            'time_delay': tau,
            'ordinal_pattern': string_pattern,
            'ordinal_probability': ordinal_probability
        })       

        if verbose:
            print(f"Procesada imagen: {image_file} - Clase: {image_class} - Método: {embedding_method}")
    
    return pd.DataFrame(result)


def ordinal_probability_plot (p, m, tau, descrip='', color='skyblue', save_path=None):
    """
    Crea un gráfico de barras para visualizar la distribución ordinal.


    Args:
        p (list): ordinal_probability. Lista de frecuencias correspondientes a cada patrón ordinal.
        m (int): Embedding dimension.
        tau (int): Time delay.
        descrip (str): Descripción adicional para el gráfico.
        color (str): Color de las barras del gráfico.

        save_path (str, optional): Ruta para guardar el gráfico. Si es None, el gráfico se muestra en pantalla.
    ---
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    labels = [i for i in range(len(p))]

    # Crear el gráfico de barras
    plt.figure(figsize=(8, 5))
    plt.bar(labels, p, color=color, edgecolor='black', alpha=0.7)
    plt.xlabel('Patrones Ordinales')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución Ordinal ($m={m}$, $\\tau={tau}$)')
    if descrip:
        plt.figtext(0.5, 0.001, descrip, wrap=True, horizontalalignment='center', fontsize=9)           
    plt.ylim(0,1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar o mostrar el gráfico
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = descrip.replace(' ', '_') + f'_m{m}_tau{tau}.png'
        plt.savefig(f"{save_path}/{file_name}")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()
                
    