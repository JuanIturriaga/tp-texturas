import brandtpompe as bp
import pandas as pd
import numpy as np
import os

dictionary_entropy_parameter = {
    'shannon': [0],
    'renyi': [0.2, 0.4, 0.8, 1, 2, 4, 8],
    'tsallis': [0.2, 0.4, 0.8, 1, 2, 4, 8]
}

for n in [64, 128]:
    for embedding in ['hilbert', 'raster1', 'raster2']:
        for m in [3,4,5,6,7]:
            for tau in [1,2,3]:
                for entropy_type in ['shannon', 'renyi', 'tsallis']:
                    for entropy_parameter in dictionary_entropy_parameter[entropy_type]:
                        entropy_parameter_str = str(entropy_parameter).replace('.','-')

                        df = pd.read_csv(f'./odp/img_data_{n}_emb_{embedding}_m{m}_tau{tau}.csv')

                        print(f'Processing {n}, {embedding}, m={m}, tau={tau}, entropy_type={entropy_type}, entropy_parameter={entropy_parameter}')

                        df_entropy_complexity = bp.entropy_complexity_save(df, entropy_type=entropy_type, entropy_parameter=entropy_parameter)

                        path_results = f'./results/{n}/{embedding}/m_{m}/tau_{tau}/'
                        file_name_results = f'entropy_complexity_{embedding}_m{m}_tau{tau}_{entropy_type}_{entropy_parameter_str}.csv'

                        if not os.path.exists(path_results):
                            os.makedirs(path_results)

                        df_entropy_complexity.to_csv(os.path.join(path_results, file_name_results), index=False)
                        print (f'Entropy and complexity results saved to {os.path.join(path_results, file_name_results)}')



