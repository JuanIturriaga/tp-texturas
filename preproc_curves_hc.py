
import os
from ordpy.ordpy import ordpy
import numpy as np


# Se crea una lista hc_maxs y hc_mins, según los posibles valores de m 
hc_maxs = [[],[],[],[],[],[],[],[]]
hc_mins = [[],[],[],[],[],[],[],[]]

for m in [3,4,5,6,7]:
    print (f'Calculating hc_max and hc_min for m={m}')
    hc_maxs[m] = ordpy.maximum_complexity_entropy(dx=m, m=100)
    hc_mins[m] = ordpy.minimum_complexity_entropy(dx=m)

print (f'hc_maxs and hc_mins created.')

# guardar en un archivo numpy
np.save('./hc_maxmin_curves.npy', [hc_maxs, hc_mins])