import brandtpompe as bp
import pandas as pd
import numpy as np
import os

limy = [0,0,0,1,1,0.2,0.03,0.015]

color_map = {
    'aluminium_foil': '#ff1493',  # Deep Pink
    'brown_bread': '#3a5fcd',  # Royal Blue
    'corduroy': '#32cd32',  # Lime Green
    'cotton': "#ffe731",  # Orange
    'cracker': '#8a2be2',  # Blue Violet
    'linen': '#00ced1',  # Dark Turquoise
    'orange_peel': "#ff9500",  # Orange Red
    'sandpaper': '#2e8b57',  # Sea Green
    'sponge': '#ff6347',  # Tomato
    'styrofoam': '#1e90ff'  # Dodger Blue   
}

main_path = 'odp'

top = 3

# diccionario para contar cuántas imágenes hay por clase
class_counts = {}
master_count = 0

for n in [64]: #[64, 128]:
    for embedding in ['hilbert']: #['hilbert', 'raster1', 'raster2']:
        for m in [7]: #[3,4,5,6,7]:            
            for tau in [1,2,3]:
                path=f'./{main_path}/{n}/{embedding}/m_{m}/tau_{tau}/'
                df = pd.read_csv(f'./{main_path}/img_data_{n}_emb_{embedding}_m{m}_tau{tau}.csv')
                for index, row in df.iterrows():
                    image_file = row['image_file']
                    image_id = row['image_id']
                    image_class = row['class']

                    class_name = image_class.replace('_','-')

                    # actualizar el contador de imágenes por clase                    
                    tag_class = f'{n}_{embedding}_m{m}_tau{tau}_{class_name}'

                    if tag_class not in class_counts:
                        class_counts[tag_class] = 0
                    class_counts[tag_class] += 1

                    if top != -1 and class_counts[tag_class] > top:
                        continue


                    master_count += 1
                    print(f'Processing {master_count} ----------------------------------------')


                    print(f'Processing {n}, {embedding}, m={m}, tau={tau}, image_id={image_id}, class={image_class}')

                    
                    
                    file_name =f'odp_{class_name}_{image_id}_m{m}_tau{tau}.npy'

                    save_path=f'./{main_path}_plot/{n}/{embedding}/m_{m}/tau_{tau}/'
                    file_name_png =f'{main_path}_plot_{class_name}_{image_id}_m{m}_tau{tau}.png'

                    descrip=f' Embedding: {embedding}, Tamaño: {n}x{n}, Imagen: {image_id}, Clase: {class_name}'

                    #cargar la distribución ordinal desde el archivo numpy
                    p = np.load(os.path.join(path, file_name), allow_pickle=True)
                    print(f'Loaded ordinal distribution from {file_name}')

                    color = color_map[image_class]
                    
                    try:
                        bp.ordinal_probability_plot (p=p, m=m, tau=tau, descrip=descrip, color=color, save_path=save_path, file_name=file_name_png, lim_y=(0,limy[m]))
                        print(f'Ordinal probability plot saved to {file_name_png}')
                    except Exception as e:
                        print(f'Error generating ordinal probability plot for {file_name_png}: {e}')

