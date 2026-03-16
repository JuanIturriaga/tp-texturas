import brandtpompe as bp
import pandas as pd

for n in [64, 128]:
    for embedding in ['hilbert', 'raster1', 'raster2']:
        df = pd.read_csv(f'./img_data_{n}_emb_{embedding}.csv')
        for m in [3,4,5,6,7]:
            for tau in [1,2,3]:
                output_path = f'./odp/{n}/{embedding}/m_{m}/tau_{tau}/'
                df_bp = bp.ordinal_distribution_save (df, m, tau,output_path)
                file_name = f'./odp/img_data_{n}_emb_{embedding}_m{m}_tau{tau}.csv'
                df_bp.to_csv(file_name, index=False)     


           