#!/usr/bin/env python3
"""
=================================================================================
TESIS UNMSM — ESCANEO χ² vs δ (Kerr-Newman carga débil)
=================================================================================

Objetivo: Encontrar el δ que minimiza χ² y cuantificar la mejora
respecto a Schwarzschild (δ=0). Esto prueba la hipótesis central de la tesis:
"La inclusión de carga débil reduce el error sistemático y mejora χ²".

Modelo: R_eff(δ) = √27 θ_g ratio_Reff(δ) con ratio_Reff exacto (no Taylor).
=================================================================================
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import least_squares

try:
    import ehtim as eh
except ImportError:
    subprocess.check_call(["pip", "install", "ehtim", "-q"])
    import ehtim as eh

# Módulo teórico de la tesis
from eht_tesis.modelo import (
    UAS,
    ratio_Reff,
    Reff_angular,
)


# ==============================================================================
# CONSTANTES
# ==============================================================================

J0_FIRST_ZERO = sp.jn_zeros(0, 1)[0]
THETA_G_SGRA = 5.03  # μas


# ==============================================================================
# MODELO DE VISIBILIDAD
# ==============================================================================

def gaussian_envelope(u_hz, w_uas):
    w_rad = w_uas * UAS
    return np.exp(-(np.pi**2 * w_rad**2 * u_hz**2) / (4.0 * np.log(2.0)))


def visibility_model(u_hz, Reff_rad, F0, w_uas):
    bessel_term = sp.j0(2.0 * np.pi * Reff_rad * u_hz)
    envelope = gaussian_envelope(u_hz, w_uas)
    return F0 * np.abs(bessel_term) * envelope


# ==============================================================================
# AJUSTE NUISANCE (F0, w) PARA δ FIJO
# ==============================================================================

def fit_nuisance_for_delta(u_hz, y, sigma, Reff_rad):
    def residuals(params):
        F0, w_uas = params
        model = visibility_model(u_hz, Reff_rad, F0, w_uas)
        return (y - model) / sigma

    p0 = [np.percentile(y, 95), 12.0]
    bounds = ([0.0, 0.1], [100.0, 100.0])
    result = least_squares(residuals, p0, bounds=bounds)

    F0_best, w_best = result.x
    model_best = visibility_model(u_hz, Reff_rad, F0_best, w_best)
    residual = (y - model_best) / sigma
    chi2 = np.sum(residual**2)
    dof = max(len(y) - 2, 1)
    return {
        "F0": F0_best, "w_uas": w_best, "model": model_best,
        "chi2": chi2, "chi2_red": chi2 / dof, "dof": dof,
        "success": result.success
    }


# ==============================================================================
# CARGAR DATOS
# ==============================================================================

def load_eht_data():
    REPO_DIR = "2024-D02-01"
    ruta = os.path.join(
        REPO_DIR, "hops_data", "April06",
        "ER6_SGRA_2017_096_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits"
    )
    obs = eh.obsdata.load_uvfits(ruta)
    data = obs.data
    u = np.asarray(data["u"], float)
    v = np.asarray(data["v"], float)
    vis = np.asarray(data["vis"], complex)
    sigma = np.asarray(data["sigma"], float)
    amp = np.abs(vis)
    u_glambda = np.sqrt(u**2 + v**2) / 1e9
    mask = (np.isfinite(u_glambda) & np.isfinite(amp) & np.isfinite(sigma)
            & (sigma > 0) & (u_glambda >= 0))
    return u_glambda[mask] * 1e9, amp[mask], sigma[mask]


# ==============================================================================
# ESCANEO PRINCIPAL
# ==============================================================================

def main():
    print("=" * 70)
    print("ESCANEO χ² vs δ — Sgr A* EHT 2024-D02-01")
    print("=" * 70)

    u_hz, y, sigma = load_eht_data()
    print(f"Datos: {len(y)} visibilidades, UV=[{u_hz.min()/1e9:.2f}, {u_hz.max()/1e9:.2f}] Gλ")

    # Rango de δ a explorar
    deltas = np.linspace(0.0, 0.4, 81)  # paso 0.005

    chi2_vals = []
    chi2_red_vals = []
    F0_vals = []
    w_vals = []

    print("\nEscaneando δ...")
    for i, d in enumerate(deltas):
        Reff_uas = Reff_angular(d, THETA_G_SGRA)
        Reff_rad = Reff_uas * UAS
        fit = fit_nuisance_for_delta(u_hz, y, sigma, Reff_rad)
        chi2_vals.append(fit["chi2"])
        chi2_red_vals.append(fit["chi2_red"])
        F0_vals.append(fit["F0"])
        w_vals.append(fit["w_uas"])
        if i % 20 == 0:
            print(f"  δ={d:.3f}: χ²={fit['chi2']:.2f}, χ²_ν={fit['chi2_red']:.3f}")

    chi2_vals = np.array(chi2_vals)
    chi2_red_vals = np.array(chi2_red_vals)

    # Mejor δ
    best_idx = np.argmin(chi2_vals)
    best_delta = deltas[best_idx]
    best_chi2 = chi2_vals[best_idx]
    best_chi2_red = chi2_red_vals[best_idx]

    chi2_schwarz = chi2_vals[0]
    chi2_red_schwarz = chi2_red_vals[0]

    delta_chi2 = best_chi2 - chi2_schwarz
    delta_chi2_red = best_chi2_red - chi2_red_schwarz

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DEL ESCANEO")
    print("=" * 70)
    print(f"χ²(δ=0)          = {chi2_schwarz:.2f}  (χ²_ν = {chi2_red_schwarz:.3f})")
    print(f"χ² mínimo        = {best_chi2:.2f}  en δ = {best_delta:.4f}")
    print(f"χ²_ν mínimo      = {best_chi2_red:.3f}")
    print(f"Δχ² (mejora)     = {delta_chi2:+.2f}")
    print(f"Δχ²_ν (mejora)   = {delta_chi2_red:+.4f}")

    if delta_chi2 < 0:
        print(f"\n>>> MEJORA ESTADÍSTICA: Kerr-Newman reduce χ² en {abs(delta_chi2):.2f}")
        print(f">>> La carga débil δ ≈ {best_delta:.4f} es preferida por los datos.")
    else:
        print(f"\n>>> NO HAY MEJORA: Schwarzschild (δ=0) es igual o mejor.")

    # Gráfica χ² vs δ
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(deltas, chi2_vals, 'b-', linewidth=1.5)
    ax.axvline(0, color='gray', linestyle=':', label='Schwarzschild (δ=0)')
    ax.axvline(best_delta, color='red', linestyle='--',
               label=f'Mínimo: δ={best_delta:.4f}')
    ax.set_xlabel('δ (parámetro de carga)')
    ax.set_ylabel('χ²')
    ax.set_title('χ² vs δ (Sgr A* EHT 2024-D02-01)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(deltas, chi2_red_vals, 'b-', linewidth=1.5)
    ax.axhline(1, color='gray', linestyle=':', label='χ²_ν = 1')
    ax.axvline(0, color='gray', linestyle=':', label='δ=0')
    ax.axvline(best_delta, color='red', linestyle='--',
               label=f'Mínimo: δ={best_delta:.4f}')
    ax.set_xlabel('δ')
    ax.set_ylabel('χ² reducido')
    ax.set_title('χ² reducido vs δ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(deltas, F0_vals, 'b-', linewidth=1.5)
    ax.axvline(0, color='gray', linestyle=':')
    ax.axvline(best_delta, color='red', linestyle='--')
    ax.set_xlabel('δ')
    ax.set_ylabel('F0 [Jy]')
    ax.set_title('Flujo ajustado vs δ')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(deltas, w_vals, 'b-', linewidth=1.5)
    ax.axvline(0, color='gray', linestyle=':')
    ax.axvline(best_delta, color='red', linestyle='--')
    ax.set_xlabel('δ')
    ax.set_ylabel('w [μas]')
    ax.set_title('Ancho gaussiano ajustado vs δ')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chi2_scan_delta.png", dpi=200, bbox_inches='tight')
    print("\n--> Gráfica guardada: chi2_scan_delta.png")
    plt.show()

    # Guardar resultados
    np.savez("chi2_scan_results.npz",
             deltas=deltas, chi2=chi2_vals, chi2_red=chi2_red_vals,
             F0=F0_vals, w=w_vals, best_delta=best_delta)
    print("--> Resultados guardados: chi2_scan_results.npz")

    # Modelo teórico: nulos vs δ
    print("\n--- PREDICCIÓN TEÓRICA DE NULOS vs δ ---")
    print("δ\t\tu₁(Gλ)\t\tΔu₁(Gλ)")
    for d in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        Reff_uas = Reff_angular(d, THETA_G_SGRA)
        u1 = J0_FIRST_ZERO / (2 * np.pi * Reff_uas * UAS) / 1e9
        Reff_uas_0 = Reff_angular(0.0, THETA_G_SGRA)
        u1_0 = J0_FIRST_ZERO / (2 * np.pi * Reff_uas_0 * UAS) / 1e9
        du1 = u1 - u1_0
        print(f"{d:.3f}\t\t{u1:.4f}\t\t{du1:+.4f}")


if __name__ == "__main__":
    main()