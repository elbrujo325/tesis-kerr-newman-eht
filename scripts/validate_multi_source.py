#!/usr/bin/env python3
"""
=================================================================================
TESIS UNMSM — VALIDACIÓN MULTI-FUENTE: M87* 2018 vs Sgr A* 2017
=================================================================================

Compara Schwarzschild vs Kerr-Newman en AMBOS objetivos EHT:
- M87* 2018 (b1, HOPS, día 111) — θ_shadow ≈ 42 μas
- Sgr A* 2017 (besttime, hi band, HOPS) — θ_shadow ≈ 50 μas

Prueba la universalidad del modelo: mismo formalismo, diferente M, D, i.
=================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import least_squares

try:
    import ehtim as eh
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "ehtim", "-q"])
    import ehtim as eh

# Módulo teórico de la tesis
from eht_tesis.modelo import (
    UAS,
    ratio_Reff,
    Reff_angular,
)


# ==============================================================================
# CONSTANTES Y MODELO
# ==============================================================================

J0_FIRST_ZERO = sp.jn_zeros(0, 1)[0]


def gaussian_envelope(u_hz, w_uas):
    w_rad = w_uas * UAS
    return np.exp(-(np.pi**2 * w_rad**2 * u_hz**2) / (4.0 * np.log(2.0)))


def visibility_model(u_hz, Reff_uas, F0, w_uas):
    """Modelo |V(u)| = F0 |J0(2π R u)| × exp(-π²w²u²/4ln2)"""
    Reff_rad = Reff_uas * UAS
    bessel_term = sp.j0(2.0 * np.pi * Reff_rad * u_hz)
    envelope = gaussian_envelope(u_hz, w_uas)
    return F0 * np.abs(bessel_term) * envelope


def fit_nuisance(u_hz, y, sigma, Reff_uas):
    """Ajusta F0, w para R_eff fijo."""
    def res(p):
        F0, w = p
        return (y - visibility_model(u_hz, Reff_uas, F0, w)) / sigma
    p0 = [np.percentile(y, 95), 12.0]
    bounds = ([0.0, 0.1], [100.0, 100.0])
    result = least_squares(res, p0, bounds=bounds)
    F0, w = result.x
    model = visibility_model(u_hz, Reff_uas, F0, w)
    chi2 = np.sum(((y - model) / sigma)**2)
    return {"F0": F0, "w": w, "model": model, "chi2": chi2,
            "chi2_red": chi2 / max(len(y)-2, 1), "success": result.success}


def load_m87_2018():
    """Carga M87* 2018 b1 HOPS día 111 desde data/ local."""
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    uvfits = os.path.join(base, "M87/2018_polarized/uvfits/"
                          "L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.uvfits")
    obs = eh.obsdata.load_uvfits(uvfits)
    data = obs.data
    u = data['u']; v = data['v']
    vis = data['vis']; sigma = data['sigma']
    amp = np.abs(vis)
    uv_gl = np.sqrt(u**2 + v**2) / 1e9
    mask = (np.isfinite(uv_gl) & np.isfinite(amp) & np.isfinite(sigma)
            & (sigma > 0) & (uv_gl >= 0.01))  # excluir baselines cortas
    return uv_gl[mask] * 1e9, amp[mask], sigma[mask], obs.rf/1e9


def load_sgra_2017():
    """Carga Sgr A* 2017 besttime hi band HOPS desde data/ local."""
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    uvfits = os.path.join(base, "SgrA/2017_first_image/uvfits_besttime/"
                          "ER6_SGRA_2017_097_hi_hops_netcal-LMTcal-besttime_StokesI.uvfits")
    obs = eh.obsdata.load_uvfits(uvfits)
    data = obs.data
    u = data['u']; v = data['v']
    vis = data['vis']; sigma = data['sigma']
    amp = np.abs(vis)
    uv_gl = np.sqrt(u**2 + v**2) / 1e9
    mask = (np.isfinite(uv_gl) & np.isfinite(amp) & np.isfinite(sigma)
            & (sigma > 0) & (uv_gl >= 0))
    return uv_gl[mask] * 1e9, amp[mask], sigma[mask], obs.rf/1e9


# ==============================================================================
# ANÁLISIS POR FUENTE
# ==============================================================================

def analyze_source(name, u_hz, y, sigma, theta_ring_uas, delta=0.1):
    print(f"\n{'='*60}")
    print(f"ANÁLISIS: {name}")
    print(f"{'='*60}")
    print(f"θ_anillo = {theta_ring_uas} μas, δ = {delta}")
    print(f"Datos: {len(y)} vis, UV=[{u_hz.min()/1e9:.2f}, {u_hz.max()/1e9:.2f}] Gλ")

    # Schwarzschild
    Reff_sch = Reff_angular(0.0, theta_ring_uas)
    fit_sch = fit_nuisance(u_hz, y, sigma, Reff_sch)

    # Kerr-Newman
    Reff_kn = Reff_angular(delta, theta_ring_uas)
    fit_kn = fit_nuisance(u_hz, y, sigma, Reff_kn)

    dchi2 = fit_kn["chi2"] - fit_sch["chi2"]
    dchi2r = fit_kn["chi2_red"] - fit_sch["chi2_red"]

    print(f"\n  Schwarzschild: χ²={fit_sch['chi2']:.1f}, χ²_ν={fit_sch['chi2_red']:.3f}")
    print(f"  Kerr-Newman:   χ²={fit_kn['chi2']:.1f}, χ²_ν={fit_kn['chi2_red']:.3f}")
    print(f"  Δχ² = {dchi2:+.1f}, Δχ²_ν = {dchi2r:+.4f}")

    # Nulos
    u1_sch = J0_FIRST_ZERO / (2*np.pi * Reff_sch * UAS) / 1e9
    u1_kn = J0_FIRST_ZERO / (2*np.pi * Reff_kn * UAS) / 1e9
    print(f"  u₁(Sch) = {u1_sch:.3f} Gλ, u₁(KN) = {u1_kn:.3f} Gλ, Δ = {u1_kn-u1_sch:+.3f} Gλ")

    return {
        "name": name, "theta_ring": theta_ring_uas,
        "sch": fit_sch, "kn": fit_kn,
        "Reff_sch": Reff_sch, "Reff_kn": Reff_kn,
        "u1_sch": u1_sch, "u1_kn": u1_kn,
        "delta_chi2": dchi2, "delta_chi2_red": dchi2r
    }


def plot_comparison(result, u_hz, y, sigma):
    """Gráfica comparativa para una fuente."""
    name = result["name"]
    u_gl = u_hz / 1e9

    u_grid = np.linspace(0.01, u_gl.max()*1.1, 1000) * 1e9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    ax.errorbar(u_gl, y, yerr=sigma, fmt='o', ms=1.5, alpha=0.2, elinewidth=0.4,
                label=f'Datos EHT ({len(y)} pts)')
    ax.plot(u_grid/1e9, visibility_model(u_grid, result["Reff_sch"],
            result["sch"]["F0"], result["sch"]["w"]), 'r-', lw=2,
            label=f'Schwarzschild (χ²_ν={result["sch"]["chi2_red"]:.2f})')
    ax.plot(u_grid/1e9, visibility_model(u_grid, result["Reff_kn"],
            result["kn"]["F0"], result["kn"]["w"]), 'b--', lw=2,
            label=f'Kerr-Newman δ=0.1 (χ²_ν={result["kn"]["chi2_red"]:.2f})')
    ax.axvline(result["u1_sch"], color='r', ls=':', alpha=0.7)
    ax.axvline(result["u1_kn"], color='b', ls=':', alpha=0.7)
    ax.set_xlabel('UV Distance [Gλ]')
    ax.set_ylabel('Amplitud [Jy]')
    ax.set_title(f'{name} — Modelo vs Datos')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, u_gl.max()*1.1)

    # Log scale
    ax = axes[1]
    ax.errorbar(u_gl, y, yerr=sigma, fmt='o', ms=1.5, alpha=0.2, elinewidth=0.4)
    ax.plot(u_grid/1e9, visibility_model(u_grid, result["Reff_sch"],
            result["sch"]["F0"], result["sch"]["w"]), 'r-', lw=2)
    ax.plot(u_grid/1e9, visibility_model(u_grid, result["Reff_kn"],
            result["kn"]["F0"], result["kn"]["w"]), 'b--', lw=2)
    ax.axvline(result["u1_sch"], color='r', ls=':', alpha=0.7)
    ax.axvline(result["u1_kn"], color='b', ls=':', alpha=0.7)
    ax.set_xlabel('UV Distance [Gλ]')
    ax.set_ylabel('Amplitud [Jy]')
    ax.set_yscale('log')
    ax.set_title(f'{name} — Escala Log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, u_gl.max()*1.1)

    plt.tight_layout()
    plt.savefig(f"comparacion_{name.replace('*','').replace(' ','_')}.png", dpi=200)
    plt.close()
    print(f"  Gráfica guardada: comparacion_{name.replace('*','').replace(' ','_')}.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("VALIDACIÓN MULTI-FUENTE: M87* vs Sgr A*")
    print("Tesis UNMSM: Kerr-Newman carga débil")
    print("=" * 70)

    # Cargar datos
    print("\nCargando M87* 2018...")
    u_m87, y_m87, sig_m87, freq_m87 = load_m87_2018()

    print("Cargando Sgr A* 2017...")
    u_sgra, y_sgra, sig_sgra, freq_sgra = load_sgra_2017()

    # Analizar ambas
    res_m87 = analyze_source("M87* 2018", u_m87, y_m87, sig_m87, theta_ring_uas=42.0)
    res_sgra = analyze_source("Sgr A* 2017", u_sgra, y_sgra, sig_sgra, theta_ring_uas=50.0)

    # Gráficas
    plot_comparison(res_m87, u_m87, y_m87, sig_m87)
    plot_comparison(res_sgra, u_sgra, y_sgra, sig_sgra)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO MULTI-FUENTE")
    print("=" * 70)
    print(f"{'Fuente':<15} {'χ²_ν(Sch)':>10} {'χ²_ν(KN)':>10} {'Δχ²_ν':>10} {'u₁(Sch)':>10} {'u₁(KN)':>10} {'Δu₁':>10}")
    print("-" * 70)
    for r in [res_m87, res_sgra]:
        print(f"{r['name']:<15} {r['sch']['chi2_red']:>10.3f} {r['kn']['chi2_red']:>10.3f} "
              f"{r['delta_chi2_red']:>+10.4f} {r['u1_sch']:>10.3f} {r['u1_kn']:>10.3f} "
              f"{r['u1_kn']-r['u1_sch']:>+10.3f}")

    print("\nConclusión:")
    for r in [res_m87, res_sgra]:
        if r["delta_chi2_red"] < 0:
            print(f"  {r['name']}: Kerr-Newman MEJORA el ajuste (Δχ²_ν = {r['delta_chi2_red']:.4f})")
        else:
            print(f"  {r['name']}: Schwarzschild igual o mejor (Δχ²_ν = {r['delta_chi2_red']:.4f})")


if __name__ == "__main__":
    main()