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
- 'odp_file': ruta y nombre del archivo numpy que contiene la distribución ordinal de la imagen. (odp: ordinal distribution pattern)
- 'entropy_type': tipo de entropía calculada (Shannon, Renyi, Tsallis).
- 'entropy_parameter' (alpha o q): parámetro utilizado para calcular la entropía de Renyi o Tsallis.
- 'entropy_value': valor numérico de la entropía normalizada calculada.
- 'complexity_value': valor numérico de la complejidad calculada.

"""


def ordinal_distribution_save (df, m=3, tau=1, output_path=None, verbose=False):
    """
    Crea y devuelve un DataFrame con la distribución ordinal.

    Args:
        df (pd.DataFrame): DataFrame que debe contener los embeddings de las imágenes y sus respectivas clases.
            Debe contener los campos 'image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method'
        m (int): Orden de los patrones ordinales.
        tau (int): Retardo temporal.
        output_path (str): Ruta para guardar el resultado de cada distribución ordinal en formato numpy. Sobreescribe los archivos si ya existen.
        verbose (bool): Si es True, muestra información adicional durante el proceso.
    ---
    Returns:
        pd.DataFrame: DataFrame con la distribución ordinal.
            Con los mismos campos de entrada y agrega: 'embedding_dimension', 'time_delay', 'odp_file'
    """
    import pandas as pd
    import os

    if output_path is None:
        raise ValueError("output_path no puede ser None. Por favor, proporciona una ruta válida para guardar los resultados.")

    if os.path.exists(output_path):
        if verbose:
            print(f"Output path {output_path} already exists. Overriding existing data.")

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
        
        # Guarda la distribución ordinal en un archivo numpy
        # odp ordinal distribution pattern
        if not os.path.exists(output_path):
            os.makedirs(output_path)            
        img_class_str = str(image_class).replace('_', '-')
        odp_file_name = f'odp_{img_class_str}_{image_id}_m{m}_tau{tau}.npy'
        odp_file = os.path.join(output_path, odp_file_name)
        #np.save(odp_file, {'patterns': ordinal_patterns, 'probabilities': ordinal_probability})
        np.save(odp_file, ordinal_probability)

        # Indica por consola que se guardó el archivo
        if verbose:
            print(f'Ordinal Distribution Pattern saved at {odp_file}')                        

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
            'odp_file': odp_file            
        })       

        if verbose:
            print(f"Procesada imagen: {image_file} - Clase: {image_class} - Método: {embedding_method}")
    
    return pd.DataFrame(result)


def entropy_complexity_save (df, entropy_type='shannon', entropy_parameter=1, verbose=False):
    """
    Realiza el cáluclo de entropia y complejidad para cada elemento del df según el método especificado por entropy_type.

    Args:
        df (pd.DataFrame): DataFrame que debe contener los embeddings de las imágenes y sus respectivas clases.
            Debe contener los campos 'image_file', 'image_id', 'image_size', 'class', 'embedding_file', 'embedding_method', 'embedding_dimension', 'time_delay', 'odp_file'
        entropy_type (str): Método de entropía a utilizar ('shannon', 'renyi', 'tsallis').
        entropy_parameter (float): Parámetro adicional para el cálculo de entropía, si es necesario.
        verbose (bool): Si es True, muestra información adicional durante el proceso.
    ---
    Returns:
        pd.DataFrame: DataFrame con la distribución ordinal.
            Con los mismos campos de entrada y agrega: 'auth', 'param', 'entropy_value', 'complexity_value'
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
        m = row['embedding_dimension']
        tau = row['time_delay']
        odp_file = row['odp_file']

        if verbose:            
            print(f"Calculando entropía y complejidad para la imagen: {image_file} - Clase: {image_class} - Método: {embedding_method}")
            print(f'Archivo cargado: {odp_file} - Tipo de entropía: {entropy_type} - Parámetro: {entropy_parameter}')

        # Carga el la distribución ordinal de la imagen
        #data = np.load(odp_file, allow_pickle=True).item()['probabilities']
        data = np.load(odp_file, allow_pickle=True)

        print(f"Distribución ordinal cargada: {data}")

        #calcula la entropía y complejidad utilizando ordpy
        if entropy_type == 'shannon':
            entropy, complexity = ordpy.complexity_entropy(data, dx=m, taux=tau, probs=True)
        elif entropy_type == 'renyi':
            entropy, complexity = ordpy.renyi_complexity_entropy(data, dx=m, taux=tau, probs=True, alpha=entropy_parameter)
        elif entropy_type == 'tsallis':
            entropy, complexity = ordpy.tsallis_complexity_entropy(data, taux=tau, probs=True, dx=m, q=entropy_parameter)

        
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
            'odp_file': odp_file, 
            'entropy_type': entropy_type,
            'entropy_parameter': entropy_parameter,
            'entropy_value': entropy,
            'complexity_value': complexity           
        })       

        if verbose:
            print(f"Procesada imagen: {image_file} - Clase: {image_class} - Método: {embedding_method}")
            print(f"Tipo de entropía: {entropy_type} - Parámetro: {entropy_parameter} - Entropía: {entropy} - Complejidad: {complexity}")
    
    return pd.DataFrame(result)


def ordinal_probability_plot (p, m, tau, descrip='', color='skyblue', save_path=None, file_name= None, lim_y=(0,1)):
    """
    Crea un gráfico de barras para visualizar la distribución ordinal.

    Args:
        p (list): ordinal_probability. Lista de frecuencias correspondientes a cada patrón ordinal.
        m (int): Embedding dimension.
        tau (int): Time delay.
        descrip (str): Descripción adicional para el gráfico.
        color (str): Color de las barras del gráfico.
        save_path (str, optional): Ruta para guardar el gráfico. Si es None, el gráfico se muestra en pantalla.
        lim ((int, int), optional): limite en y al que se debe adaptar el gráfico.
    ---
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    
    labels = [i for i in range(len(p))]

    #recorrer p de atrás hacia adelante hasta que sea distinto de cero
    i = len(p)-1
    while (i>0 and p[i]==0):
        i -= 1

    # Crear el gráfico de barras
    plt.figure(figsize=(8, 5))
    plt.bar(labels, p, color=color, edgecolor='black', alpha=0.7)
    plt.xlabel('Patrones Ordinales')
    plt.ylabel('Frecuencia')
    plt.title(f'Distribución Ordinal ($m={m}$, $\\tau={tau}$)')
    if descrip:
        plt.figtext(0.5, 0.001, descrip, wrap=True, horizontalalignment='center', fontsize=9)           
    plt.ylim(lim_y[0],lim_y[1])
    plt.xlim(0-0.5, i+0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Guardar o mostrar el gráfico
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if file_name is None:
            file_name = descrip.replace(' ', '_') + f'_m{m}_tau{tau}.png'
        plt.savefig(f"{save_path}/{file_name}")
        print(f"Gráfico guardado en: {save_path}")
    else:
        plt.show()


def entropy_complexity_plot (df, descrip=None, color_map=None, save_path=None, file_name= None, lim=(0,1), hc_max=[], hc_min=[], classes=[], images=[], top=-1, legend=True, xlim=(0,1), ylim=(0,1)):
    """
    Crea un gráfico de dispersión para visualizar la relación entre entropía y complejidad.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas: 'image_id', 'class', 'embedding_method', 'embedding_dimension', 
                          'time_delay', 'entropy_type', 'entropy_parameter', 'entropy_value', 'complexity_value'.
        descrip (str, optional): Descripción adicional para el gráfico.
        color_map (list, optional): Lista de colores para cada punto en el gráfico. Si es None, se utilizará un color predeterminado.
        save_path (str, optional): Ruta para guardar el gráfico. Si es None, el gráfico se muestra en pantalla.
    ---
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import util as u

    # Valores predeterminados para m y tau con el primer registro del DataFrame, si no se proporcionan explícitamente
    m = df['embedding_dimension'].iloc[0] if 'embedding_dimension' in df.columns else 3
    tau = df['time_delay'].iloc[0] if 'time_delay' in df.columns else 1

    # Obtener las curvas de límite del plano usando ordpy
    # Estas funciones devuelven arreglos 2D donde:
    # [:, 0] corresponde al eje X (Entropía H)
    # [:, 1] corresponde al eje Y (Complejidad C)
    if (len(hc_max) == 0):
        hc_max = ordpy.maximum_complexity_entropy(dx=m, m=100)

    if (len(hc_min) == 0):
        hc_min = ordpy.minimum_complexity_entropy(dx=m)

    # Graficar el Plano de Entropía-Complejidad
    plt.figure(figsize=(8, 6))

    # Dibujar las curvas límite de ordpy
    if legend:
        plt.plot(hc_max[:, 0], hc_max[:, 1], color="#ff1493", linewidth=1.5, label='Límite Máximo')
        plt.plot(hc_min[:, 0], hc_min[:, 1], color="#3a5fcd", linewidth=1.5, label='Límite Mínimo')
    else:
        plt.plot(hc_max[:, 0], hc_max[:, 1], color="#ff1493", linewidth=1.5)
        plt.plot(hc_min[:, 0], hc_min[:, 1], color="#3a5fcd", linewidth=1.5)


    # Registro de clases impresas para mostrar en la leyenda
    clases_print = []
    cont_classes = len(classes) #las clases a mostrar
    cont_images = len(images) #los ejemplos a mostrar

    # Obtiene todas las clases distintas del df
    df_classes = df['class'].unique()

    # Crea un diccionario por cada clase para contar el top mostrado
    top_mostrado = {clase: 0 for clase in df_classes}


    entropy_type = ''
    entropy_parameter = 0
    
    # Itera sobre cada fila del DataFrame de entrada
    for index, row in df.iterrows():
        image_id = row['image_id']
        image_size = row['image_size']
        image_class = row['class']
        embedding_method = row['embedding_method']
        entropy_type = row['entropy_type']
        entropy_parameter = row['entropy_parameter']
        entropy = row['entropy_value']
        complexity = row['complexity_value']

        # Verificar si la fila cumple con los criterios de m y tau
        if m != row['embedding_dimension']:
            continue

        if tau != row['time_delay']:
            continue

        # Verificar si la clase y el ejemplo cumplen con los criterios para ser graficados
        if (cont_classes == 0 or (image_class in classes)) and (cont_images == 0 or (image_id in images)) and (top<0 or top_mostrado[image_class] < top):
            # Asignar un color específico para cada clase
            if color_map and image_class in color_map:
                color = color_map[image_class]
            else:
                color = 'skyblue'  # Color predeterminado

            # calcular edgecolor 
            edgecolor = u.oscurecer_color(color, 0.5)

            # Graficar el punto en el plano de entropía-complejidad
            plt.scatter(entropy, complexity, edgecolor=edgecolor, color=color, alpha=0.7)

            # Agregar la clase a la leyenda si no ha sido registrada
            if legend and image_class not in clases_print:
                plt.scatter([], [], color=color, alpha=0.7, edgecolor=edgecolor,label=image_class) 
                clases_print.append(image_class)

            # Incrementar el contador de ejemplos mostrados para la clase
            top_mostrado[image_class] += 1

    # Configuraciones estéticas
    if entropy_type == 'shannon':
        plt.title(f'Plano de Entropía-Complejidad {entropy_type} ($m={m}$, $\\tau={tau}$)', fontsize=14)
    else:
        plt.title(f'Plano de Entropía-Complejidad {entropy_type} (q={entropy_parameter}, $m={m}$, $\\tau={tau}$)', fontsize=14)
        
    plt.xlabel('Entropía de Permutación ($H$)', fontsize=12)
    plt.ylabel('Complejidad Estadística ($C$)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    if descrip:
        plt.figtext(0.5, 0.001, descrip, wrap=True, horizontalalignment='center', fontsize=9)  

    # Ajustar límites de los ejes (opcional, el plano estándar suele ir de 0 a 1 en H)
    plt.xlim(xlim)
    plt.ylim(ylim)

     # Guardar o mostrar el gráfico
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, file_name)
        print(f'Guardando figura en: {file_path}')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    