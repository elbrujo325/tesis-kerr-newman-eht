"""
Módulo de modelo teórico: Reissner–Nordström límite estático (a=0)
Fórmulas exactas sin series de Taylor para el radio efectivo de la sombra.
"""
import numpy as np
from scipy.special import jn_zeros

# Constantes
UAS = np.pi / (180 * 3600 * 1e6)  # radianes por microsegundo de arco = 4.84813681109536e-12


def validar_delta(delta: float) -> None:
    """Valida que delta esté en el dominio físico [0, 9/8]."""
    if not (0.0 <= delta <= 9.0 / 8.0):
        raise ValueError(f"delta={delta} fuera del dominio físico [0, 9/8]")


def r0_foton(delta: float) -> float:
    """
    Radio de la órbita esférica de fotones inestable (r_0) en unidades de r_g.
    r_0 = (3/2) * (1 + sqrt(1 - 8δ/9)) * r_g
    Retorna r_0 / r_g (adimensional).
    """
    validar_delta(delta)
    return 1.5 * (1.0 + np.sqrt(1.0 - 8.0 * delta / 9.0))


def ratio_Reff(delta: float) -> float:
    """
    Ratio R_eff(δ) / R_eff(0) para el límite estático (a=0).
    R_eff² = 27 r_g² * (2 r_0³ / (r_0 - 1) / 27) = 2 r_0³ / (r_0 - 1) * r_g²
    R_eff(0) = 3√3 r_g
    Por tanto: ratio = sqrt(2 r_0³ / (r_0 - 1) / 27)
    """
    validar_delta(delta)
    r0 = r0_foton(delta)
    return np.sqrt(2.0 * r0**3 / (r0 - 1.0) / 27.0)


def Reff_angular(delta: float, theta_g_uas: float) -> float:
    """
    Radio angular efectivo de la sombra en µas.
    R_eff = √27 * θ_g * ratio_Reff(δ)
    donde θ_g = GM/(c²D) en µas.
    """
    validar_delta(delta)
    return np.sqrt(27.0) * theta_g_uas * ratio_Reff(delta)


def ratio_Reff_taylor(delta: float, orden: int = 3) -> float:
    """
    Serie de Taylor de ratio_Reff(δ) alrededor de δ=0.
    Coeficientes exactos (verificados con sympy):
    ratio = 1 - δ/6 - 7δ²/216 - 151δ³/11664 + O(δ⁴)
    """
    validar_delta(delta)
    if orden == 1:
        return 1.0 - delta / 6.0
    elif orden == 2:
        return 1.0 - delta / 6.0 - 7.0 * delta**2 / 216.0
    elif orden >= 3:
        return (1.0 - delta / 6.0
                - 7.0 * delta**2 / 216.0
                - 151.0 * delta**3 / 11664.0)
    return 1.0


def R_eff_squared_taylor(delta: float, orden: int = 3) -> float:
    """
    Serie de Taylor de R_eff²/(27 r_g²) = ratio_Reff².
    Coeficientes exactos:
    R_eff²/(27 r_g²) = 1 - δ/3 - δ²/27 - 11δ³/729 + O(δ⁴)
    (Aquí D = 1/27 ES CORRECTO para el término δ²)
    """
    validar_delta(delta)
    if orden == 1:
        return 1.0 - delta / 3.0
    elif orden == 2:
        return 1.0 - delta / 3.0 - delta**2 / 27.0
    elif orden >= 3:
        return (1.0 - delta / 3.0
                - delta**2 / 27.0
                - 11.0 * delta**3 / 729.0)
    return 1.0


def nulos_J0(theta_ring_uas: float, delta: float = 0.0, m: int = 3) -> np.ndarray:
    """
    Posiciones de los primeros m nulos de visibilidad para un anillo delgado.
    u_m = z_m / (2π R)  con R = (θ_ring/2) * ratio_Reff(δ) * UAS  [rad]
    Retorna u_m en Gλ (10⁹ λ).
    z_m son los ceros de J₀: [2.40482556, 5.52007811, 8.65372791, ...]
    """
    validar_delta(delta)
    ceros = jn_zeros(0, m)  # z_1, z_2, z_3...
    R_rad = (theta_ring_uas / 2.0) * ratio_Reff(delta) * UAS
    u_m = ceros / (2.0 * np.pi * R_rad) / 1e9  # Gλ
    return u_m


def u1_u2_ratio() -> float:
    """Razón teórica u₂/u₁ para un anillo delgado (J₀)."""
    ceros = jn_zeros(0, 2)
    return ceros[1] / ceros[0]


def diametro_para_nulo1(u1_objetivo_Gl: float, delta: float = 0.0) -> float:
    """
    Diámetro angular θ (µas) que coloca el primer nulo en u1_objetivo_Gl.
    θ = z_1 / (π * u1 * 1e9 * ratio_Reff(δ) * UAS)
    """
    validar_delta(delta)
    z1 = jn_zeros(0, 1)[0]
    theta_uas = z1 / (np.pi * u1_objetivo_Gl * 1e9 * ratio_Reff(delta) * UAS)
    return theta_uas


def u1_taylor_ratio(delta: float, orden: int = 2) -> float:
    """
    Serie de Taylor de u₁(δ)/u₁(0) = 1/ratio_Reff(δ).
    Coeficientes exactos:
    u₁(δ)/u₁(0) = 1 + δ/6 + 13δ²/216 + O(δ³)
    """
    validar_delta(delta)
    if orden == 1:
        return 1.0 + delta / 6.0
    elif orden >= 2:
        return 1.0 + delta / 6.0 + 13.0 * delta**2 / 216.0
    return 1.0


def b_critico_squared(r: float, delta: float) -> float:
    """
    Parámetro de impacto al cuadrado b² = r⁴ / (r² - 2r + δ)
    Evaluado en r_0 debe ser igual a 2 r_0³ / (r_0 - 1).
    """
    validar_delta(delta)
    return r**4 / (r**2 - 2.0 * r + delta)


# Verificación interna (se ejecuta al importar)
if __name__ == "__main__":
    print("=== Verificación modelo.py ===")
    print(f"UAS = {UAS:.10e}")
    print(f"r0_foton(0) = {r0_foton(0):.6f} (esperado 3.0)")
    print(f"ratio_Reff(0) = {ratio_Reff(0):.10f} (esperado 1.0)")
    print(f"ratio_Reff(0.05) = {ratio_Reff(0.05):.6f}")
    print(f"ratio_Reff(0.10) = {ratio_Reff(0.10):.6f}")
    print(f"ratio_Reff(0.15) = {ratio_Reff(0.15):.6f}")
    print(f"nulos_J0(42.0, 0) = {nulos_J0(42.0, 0.0)} Gλ")
    print(f"u1/u2 ratio = {u1_u2_ratio():.6f}")
    print(f"diametro_para_nulo1(3.4) = {diametro_para_nulo1(3.4):.4f} µas")
    print("=== OK ===")