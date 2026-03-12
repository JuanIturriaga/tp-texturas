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
- 'entropy_type': tipo de entropía calculada (Shannon, Renyi, Tsallis).
- 'entropy_parameter' (alpha o q): parámetro utilizado para calcular la entropía de Renyi o Tsallis.
- 'entropy_value': valor numérico de la entropía normalizada calculada.
- 'complexity_value': valor numérico de la complejidad calculada.

"""


def ordinal_distribution_save (df, m=3, tau=1, verbose=False):
    """
    Guarda la distribución ordinal en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas 'Ordinal Pattern' y 'Frequency'.
        m (int): Orden de los patrones ordinales.
        tau (int): Retardo temporal.
        verbose (bool): Si es True, muestra información adicional durante el proceso.

    Returns:
        None
    """
    import pandas as pd

    # Convertir el diccionario a un DataFrame de pandas
    df = pd.DataFrame(list(data.items()), columns=['Ordinal Pattern', 'Frequency'])

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"Ordinal distribution saved to: {output_path}")

