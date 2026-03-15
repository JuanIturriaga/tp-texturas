
def oscurecer_color(hex_color, factor=0.5):
    """Oscurece un color hexadecimal dado un factor (0-1)."""
    # Eliminar '#' si existe
    hex_color = hex_color.lstrip('#')
    
    # Convertir hex a RGB
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Aplicar factor de oscurecimiento
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    
    # Asegurar que no sean negativos y volver a hex
    return '#{:02x}{:02x}{:02x}'.format(
        max(0, r), max(0, g), max(0, b)
    )