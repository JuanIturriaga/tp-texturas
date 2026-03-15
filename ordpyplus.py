import numpy as np
import math

def _renyi_entropy_prob(p, q):
    """Calcula la entropía de Rényi para una distribución de probabilidad."""
    p = p[p > 0]  # Evitar logaritmos de cero
    if q == 1.0:
        return -np.sum(p * np.log(p))
    return (1 / (1 - q)) * np.log(np.sum(p ** q))

def _renyi_complexity_entropy_prob(p, q, N):
    """Calcula la Entropía (H) y Complejidad (C) de Rényi teóricas para una distribución p."""
    U = np.ones(N) / N  # Distribución uniforme
    
    # 1. Entropía Normalizada (H)
    Sq_p = _renyi_entropy_prob(p, q)
    H_norm = Sq_p / np.log(N)
    
    # 2. Desequilibrio (Q) usando Divergencia de Jensen-Rényi
    P_plus_U_half = (p + U) / 2
    Sq_mix = _renyi_entropy_prob(P_plus_U_half, q)
    Sq_U = _renyi_entropy_prob(U, q)
    Jq = Sq_mix - 0.5 * Sq_p - 0.5 * Sq_U
    
    # Máxima Divergencia (ocurre con distribución Delta)
    P_delta = np.zeros(N)
    P_delta[0] = 1.0
    P_delta_plus_U_half = (P_delta + U) / 2
    Sq_mix_max = _renyi_entropy_prob(P_delta_plus_U_half, q)
    Sq_delta = _renyi_entropy_prob(P_delta, q)  # Teóricamente 0
    Jq_max = Sq_mix_max - 0.5 * Sq_delta - 0.5 * Sq_U
    
    # Normalización del desequilibrio
    Q = Jq / Jq_max if Jq_max > 0 else 0
    
    # 3. Complejidad Estadística (C)
    C = Q * H_norm
    
    return H_norm, C

def maximum_complexity_entropy_renyi(dx=3, q=0.2, num_points=200):
    """Genera la curva del límite máximo para el plano de Rényi."""
    N = math.factorial(dx)
    H_vals, C_vals = [], []
    
    # La curva máxima conecta mezclas de k y k+1 estados equiprobables
    for k in range(1, N):
        p_start = 1 / (k + 1)
        p_end = 1 / k
        p_vals = np.linspace(p_start, p_end, num_points)
        
        for p in p_vals:
            P = np.zeros(N)
            P[:k] = p
            P[k] = 1 - k * p
            
            H, C = _renyi_complexity_entropy_prob(P, q, N)
            H_vals.append(H)
            C_vals.append(C)
            
    curve = np.array(list(zip(H_vals, C_vals)))
    return curve[curve[:, 0].argsort()]  # Ordenar por H para graficar correctamente

def minimum_complexity_entropy_renyi(dx=3, q=0.2, num_points=1000):
    """Genera la curva del límite mínimo para el plano de Rényi."""
    N = math.factorial(dx)
    H_vals, C_vals = [], []
    
    # La curva mínima mezcla 1 estado dominante con los demás equiprobables
    p_vals = np.linspace(1/N, 1, num_points)
    for p in p_vals:
        P = np.ones(N) * ((1 - p) / (N - 1))
        P[0] = p
        
        H, C = _renyi_complexity_entropy_prob(P, q, N)
        H_vals.append(H)
        C_vals.append(C)
        
    curve = np.array(list(zip(H_vals, C_vals)))
    return curve[curve[:, 0].argsort()]



def _tsallis_entropy_prob(p, q):
    """Calcula la entropía de Tsallis para una distribución de probabilidad."""
    p = p[p > 0]  # Evitar logaritmos y potencias problemáticas con ceros
    if q == 1.0:
        return -np.sum(p * np.log(p))  # Límite de Shannon cuando q -> 1
    return (1 / (q - 1)) * (1 - np.sum(p ** q))

def _tsallis_complexity_entropy_prob(p, q, N):
    """Calcula la Entropía (H) y Complejidad (C) de Tsallis teóricas para una distribución p."""
    U = np.ones(N) / N  # Distribución uniforme
    
    # 1. Entropía Normalizada (H)
    Sq_p = _tsallis_entropy_prob(p, q)
    Sq_U = _tsallis_entropy_prob(U, q)
    H_norm = Sq_p / Sq_U if Sq_U > 0 else 0
    
    # 2. Desequilibrio (Q) usando Divergencia de Jensen-Tsallis
    P_plus_U_half = (p + U) / 2
    Sq_mix = _tsallis_entropy_prob(P_plus_U_half, q)
    Jq = Sq_mix - 0.5 * Sq_p - 0.5 * Sq_U
    
    # Máxima Divergencia (ocurre con distribución Delta)
    P_delta = np.zeros(N)
    P_delta[0] = 1.0
    P_delta_plus_U_half = (P_delta + U) / 2
    Sq_mix_max = _tsallis_entropy_prob(P_delta_plus_U_half, q)
    Sq_delta = _tsallis_entropy_prob(P_delta, q)  # Teóricamente 0
    Jq_max = Sq_mix_max - 0.5 * Sq_delta - 0.5 * Sq_U
    
    # Normalización del desequilibrio
    Q = Jq / Jq_max if Jq_max > 0 else 0
    
    # 3. Complejidad Estadística (C)
    C = Q * H_norm
    
    return H_norm, C

def maximum_complexity_entropy_tsallis(dx=3, q=0.2, num_points=200):
    """Genera la curva del límite máximo para el plano de Tsallis."""
    N = math.factorial(dx)
    H_vals, C_vals = [], []
    
    # La curva máxima conecta mezclas de k y k+1 estados equiprobables
    for k in range(1, N):
        p_start = 1 / (k + 1)
        p_end = 1 / k
        p_vals = np.linspace(p_start, p_end, num_points)
        
        for p in p_vals:
            P = np.zeros(N)
            P[:k] = p
            P[k] = 1 - k * p
            
            H, C = _tsallis_complexity_entropy_prob(P, q, N)
            H_vals.append(H)
            C_vals.append(C)
            
    curve = np.array(list(zip(H_vals, C_vals)))
    return curve[curve[:, 0].argsort()]  # Ordenar por H para ploteo continuo

def minimum_complexity_entropy_tsallis(dx=3, q=0.2, num_points=1000):
    """Genera la curva del límite mínimo para el plano de Tsallis."""
    N = math.factorial(dx)
    H_vals, C_vals = [], []
    
    # La curva mínima mezcla 1 estado dominante con los demás equiprobables
    p_vals = np.linspace(1/N, 1, num_points)
    for p in p_vals:
        P = np.ones(N) * ((1 - p) / (N - 1))
        P[0] = p
        
        H, C = _tsallis_complexity_entropy_prob(P, q, N)
        H_vals.append(H)
        C_vals.append(C)
        
    curve = np.array(list(zip(H_vals, C_vals)))
    return curve[curve[:, 0].argsort()]