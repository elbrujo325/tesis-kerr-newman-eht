"""
Módulo de detectabilidad: análisis honesto de sensibilidad a δ.
"""
import numpy as np
from scipy.optimize import brentq
from eht_tesis.modelo import ratio_Reff, diametro_para_nulo1


def delta_relativo(delta):
    """ΔR/R = 1 - ratio_Reff(δ)"""
    return 1.0 - ratio_Reff(delta)


def delta_R_uas(delta, theta_ring_uas=42.0):
    """ΔR en µas para un diámetro de anillo de referencia."""
    R0 = theta_ring_uas / 2.0
    return R0 * (1.0 - ratio_Reff(delta))


def delta_u1_Gl(delta, theta_ring_uas=42.0):
    """Δu₁ en Gλ para un diámetro de anillo de referencia."""
    from eht_tesis.modelo import nulos_J0
    u1_0 = nulos_J0(theta_ring_uas, 0.0)[0]
    u1_d = nulos_J0(theta_ring_uas, delta)[0]
    return u1_d - u1_0


def sigma_rel_theta_g():
    """
    Incertidumbre relativa en θ_g = GM/(c²D).
    M ≈ 6.5±0.7×10⁹ M☉ (EHT 2019, Paper I) → ±10.8%
    D ≈ 16.8±0.8 Mpc → ±4.8%
    Sumadas en cuadratura: sqrt(10.8² + 4.8²) ≈ 11.8%
    """
    sigma_M_rel = 0.7 / 6.5  # ≈ 0.1077
    sigma_D_rel = 0.8 / 16.8  # ≈ 0.0476
    return np.sqrt(sigma_M_rel**2 + sigma_D_rel**2)  # ≈ 0.118


def sigma_rel_diametro():
    """
    Incertidumbre en diámetro del anillo ±3 µas (EHT 2019, Paper IV).
    Para θ = 42 µas: 3/42 ≈ 7.1%
    """
    return 3.0 / 42.0  # ≈ 0.0714


def sigma_rel_ancho_minimo(uv_Gl, amp, sigma):
    """
    Incertidumbre efectiva desde el ancho observado del mínimo de visibilidad.
    Aproximación: σ_u1 ≈ (ancho del mínimo) / SNR_efectivo
    """
    # Buscar región del mínimo (3.3-4.2 Gλ)
    mask = (uv_Gl > 3.3) & (uv_Gl < 4.2)
    if mask.sum() < 10:
        return 0.1  # fallback
    # Ancho aproximado del mínimo
    u_min = uv_Gl[mask]
    a_min = amp[mask]
    # Encontrar mínimo local
    idx_min = np.argmin(a_min)
    # Ancho a mitad de altura (muy aproximado)
    half_max = (a_min.max() + a_min.min()) / 2
    # Simplificación: usar dispersión de uv en la región
    sigma_u = u_min.std()
    snr_eff = a_min.mean() / sigma[mask].mean()
    return sigma_u / snr_eff if snr_eff > 0 else 0.1


def delta_minimo_detectable(sigma_rel, delta_max=0.4):
    """
    Menor δ tal que ΔR/R ≥ 3·σ_rel.
    Usa brentq para resolver 1 - ratio_Reff(δ) = 3*sigma_rel.
    """
    if sigma_rel <= 0:
        return 0.0
    target = 3.0 * sigma_rel

    def f(delta):
        return 1.0 - ratio_Reff(delta) - target

    # Verificar que hay solución en [0, delta_max]
    if f(delta_max) < 0:
        return delta_max  # no detectable en rango
    if f(0.0) > 0:
        return 0.0  # detectable incluso en δ=0 (no físico)

    try:
        delta_min = brentq(f, 0.0, delta_max)
        return delta_min
    except ValueError:
        return delta_max


def precision_MD_para_delta(delta_target, sigma_rel_current):
    """
    Precisión requerida en M/D para detectar δ_target a 3σ.
    sigma_rel_requerida = (1 - ratio_Reff(delta_target)) / 3
    Como sigma_rel ≈ sqrt((σM/M)² + (σD/D)²), asumiendo σM/M ≈ σD/D:
    precision_requerida = sigma_rel_requerida / sqrt(2)
    """
    delta_R_rel = 1.0 - ratio_Reff(delta_target)
    sigma_rel_req = delta_R_rel / 3.0
    precision_MD = sigma_rel_req / np.sqrt(2)
    return precision_MD


def tabla_detectabilidad(deltas, theta_ring_uas=42.0, uv_Gl=None, amp=None, sigma=None):
    """
    Produce tabla de detectabilidad para lista de deltas.
    """
    sigma_theta_g = sigma_rel_theta_g()
    sigma_diam = sigma_rel_diametro()
    sigma_min = sigma_rel_ancho_minimo(uv_Gl, amp, sigma) if uv_Gl is not None else 0.1

    print(f"σ_rel(θ_g) = {sigma_theta_g:.3f} ({100*sigma_theta_g:.1f}%)")
    print(f"σ_rel(diám.) = {sigma_diam:.3f} ({100*sigma_diam:.1f}%)")
    print(f"σ_rel(min.) = {sigma_min:.3f} ({100*sigma_min:.1f}%)\n")

    print("δ\tΔR/R\tΔR (µas)\tΔu₁ (Gλ)\t3σ_θg\t3σ_diam\t3σ_min")
    for d in deltas:
        dR_R = delta_relativo(d)
        dR = delta_R_uas(d, theta_ring_uas)
        du1 = delta_u1_Gl(d, theta_ring_uas)
        print(f"{d:.3f}\t{dR_R:.4f}\t{dR:.2f}\t\t{du1:.4f}\t"
              f"{3*sigma_theta_g:.4f}\t{3*sigma_diam:.4f}\t{3*sigma_min:.4f}")

    # δ mínimos detectables
    print("\nδ mínimo detectable a 3σ:")
    for label, sig in [("θ_g prior", sigma_theta_g), ("diám. anillo", sigma_diam), ("ancho mínimo", sigma_min)]:
        dmin = delta_minimo_detectable(sig)
        print(f"  {label}: δ_min = {dmin:.3f}")

    # Precisión M/D requerida
    print("\nPrecisión relativa requerida en M/D para detectar a 3σ:")
    for d in [0.05, 0.10]:
        prec = precision_MD_para_delta(d, sigma_theta_g)
        print(f"  δ={d:.2f}: σ_M/M ≈ σ_D/D ≤ {prec:.3f} ({100*prec:.1f}%)")

    return {
        "sigma_theta_g": sigma_theta_g,
        "sigma_diametro": sigma_diam,
        "sigma_minimo": sigma_min,
    }


if __name__ == "__main__":
    deltas_test = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    tabla_detectabilidad(deltas_test)