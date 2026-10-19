"""
eht_tesis - Paquete para análisis de datos EHT en tesis Kerr-Newman carga débil
"""
from .modelo import (
    UAS,
    r0_foton,
    ratio_Reff,
    ratio_Reff_taylor,
    nulos_J0,
    diametro_para_nulo1,
    validar_delta,
)

__all__ = [
    "UAS",
    "r0_foton",
    "ratio_Reff",
    "ratio_Reff_taylor",
    "nulos_J0",
    "diametro_para_nulo1",
    "validar_delta",
]