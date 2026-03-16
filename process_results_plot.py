import brandtpompe as bp
import pandas as pd
import os
import numpy as np
from ordpy.ordpy import ordpy


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

dictionary_entropy_parameter = {
    'shannon': [0],
    'renyi': [0.2, 0.4, 0.8, 1, 2, 4, 8],
    'tsallis': [0.2, 0.4, 0.8, 1, 2, 4, 8]
}

# Cargar las curvas hc_max y hc_min desde el archivo numpy
hc_maxmin_curves = np.load('./hc_maxmin_curves.npy', allow_pickle=True)

hc_maxs = hc_maxmin_curves[0]
hc_mins = hc_maxmin_curves[1]

classes=[] #['cotton', 'orange_peel']  # Classes a mostrar en el gráfico
top = -1
xlim=(0,1)
ylim=(0,1)
main_path = 'results'

for n in [128]: #for n in [64, 128]:
    for embedding in ['hilbert', 'raster1', 'raster2']:
        for m in [3,4,5,6,7]:
            for tau in [1,2,3]:
                for entropy_type in ['shannon', 'renyi', 'tsallis']:
                    for entropy_parameter in dictionary_entropy_parameter[entropy_type]:
                        entropy_parameter_str = str(entropy_parameter).replace('.','-')

                        path=f'./{main_path}/{n}/{embedding}/m_{m}/tau_{tau}/'
                        file_name = f'entropy_complexity_{embedding}_m{m}_tau{tau}_{entropy_type}_{entropy_parameter_str}.csv'

                        curve_path = f'./{main_path}_plot/{n}/{embedding}/m_{m}/tau_{tau}/'
                        curve_file_name = f'entropy_complexity_{embedding}_m{m}_tau{tau}_{entropy_type}_{entropy_parameter_str}.png'

                        df = pd.read_csv(os.path.join(path, file_name))

                        print(f'Processing {n}, {embedding}, m={m}, tau={tau}, entropy_type={entropy_type}, entropy_parameter={entropy_parameter}')

                        descrip = f'Embedding: {embedding}, Tamaño: {n}x{n}'

                        bp.entropy_complexity_plot (df, descrip=descrip, color_map=color_map, save_path=curve_path, file_name= curve_file_name, lim=(0,1), hc_max=hc_maxs[m], hc_min=hc_mins[m], classes=classes, images=[], top=top, legend=True, xlim=xlim, ylim=ylim)

