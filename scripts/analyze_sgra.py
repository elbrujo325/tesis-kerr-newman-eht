#!/usr/bin/env python3
"""
=================================================================================
TESIS UNMSM — ANÁLISIS DE DATOS EHT 2024-D02-01 — Sgr A*
=================================================================================

Modelo: Schwarzschild vs Kerr-Newman débilmente cargado
Observable: Amplitud de visibilidad de Stokes I |V(u,v)|
Modelo efectivo de anillo fino:
    |V(u)| = F0 |J0(2π R_eff u)| × E(u)
donde:
    E(u) = exp[-π² w² u² / (4 ln 2)]

IMPORTANTE:
Este análisis utiliza el modelo efectivo radial de la tesis.
No pretende sustituir todavía el ray-tracing Kerr-Newman 2D completo.
=================================================================================
"""

# ==============================================================================
# 0. LIBRERÍAS
# ==============================================================================
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import least_squares

# EHTIM
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
    nulos_J0,
)


# ==============================================================================
# 1. CONSTANTES FÍSICAS
# ==============================================================================

# Primer cero de J0
J0_FIRST_ZERO = sp.jn_zeros(0, 1)[0]  # 3.831705970...

# Escala gravitacional angular de Sgr A* (EHT 2022, Paper I)
THETA_G_SGRA = 5.03  # μas

# Parámetro de carga débil para comparativa fija
DELTA = 0.1


# ==============================================================================
# 2. CARGAR DATOS EHT OFICIALES (REPOSITORIO 2024-D02-01)
# ==============================================================================

REPO_URL = "https://github.com/eventhorizontelescope/2024-D02-01.git"
REPO_DIR = "2024-D02-01"


def cargar_datos_eht():
    """Clona el repositorio oficial y carga el dataset Sgr A* 6 abril 2017."""

    # Clonar repo si no existe
    if not os.path.isdir(REPO_DIR):
        print("--> Clonando repositorio oficial EHT 2024-D02-01...")
        subprocess.run(
            ["git", "clone", REPO_URL, REPO_DIR],
            check=True
        )
    else:
        print("--> El repositorio 2024-D02-01 ya existe.")

    # Dataset: 6 abril 2017, banda alta, HOPS, polarimétrico calibrado
    ruta_datos = os.path.join(
        REPO_DIR,
        "hops_data",
        "April06",
        "ER6_SGRA_2017_096_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits"
    )

    if not os.path.isfile(ruta_datos):
        raise FileNotFoundError(
            f"No se encontró el dataset oficial:\n{ruta_datos}"
        )

    print("\n--> Dataset EHT seleccionado:")
    print(ruta_datos)

    # Cargar UVFITS
    print("\n--> Cargando UVFITS con eht-imaging...")
    obs = eh.obsdata.load_uvfits(ruta_datos)

    print("\n--- OBSERVACIÓN CARGADA ---")
    print(obs)

    # Extraer datos
    data = obs.data

    # Verificar campos requeridos
    required_fields = ["vis", "sigma", "u", "v"]
    for field in required_fields:
        if field not in data.dtype.names:
            raise RuntimeError(f"El UVFITS no contiene el campo '{field}' esperado.")

    u_raw = np.asarray(data["u"], dtype=float)
    v_raw = np.asarray(data["v"], dtype=float)
    vis_complex = np.asarray(data["vis"], dtype=complex)
    sigma_raw = np.asarray(data["sigma"], dtype=float)

    return u_raw, v_raw, vis_complex, sigma_raw, obs


# ==============================================================================
# 3. PREPROCESAMIENTO DE VISIBILIDADES
# ==============================================================================

def preprocesar_visibilidades(u_raw, v_raw, vis_complex, sigma_raw):
    """Limpieza básica y cálculo de magnitudes."""

    vis_amp = np.abs(vis_complex)
    u_glambda = np.sqrt(u_raw**2 + v_raw**2) / 1e9  # Gλ

    # Máscara de limpieza
    mask = (
        np.isfinite(u_glambda)
        & np.isfinite(vis_amp)
        & np.isfinite(sigma_raw)
        & (sigma_raw > 0)
        & (u_glambda >= 0)
    )

    u_glambda = u_glambda[mask]
    vis_amp = vis_amp[mask]
    sigma_raw = sigma_raw[mask]

    print(f"\n--> Datos válidos: {len(vis_amp):,} visibilidades")
    print(f"    UV range: [{u_glambda.min():.2f}, {u_glambda.max():.2f}] Gλ")
    print(f"    Amplitud: [{vis_amp.min():.4f}, {vis_amp.max():.4f}] Jy")
    print(f"    Sigma: [{sigma_raw.min():.4f}, {sigma_raw.max():.4f}] Jy")

    return u_glambda, vis_amp, sigma_raw


# ==============================================================================
# 4. MODELO DE VISIBILIDAD (ANILLO FINO + ENVOLVENTE GAUSSIANA)
# ==============================================================================

def gaussian_envelope(u_hz, w_uas):
    """
    Envolvente gaussiana circular.
    w_uas: escala angular en microsegundos de arco.
    """
    w_rad = w_uas * UAS
    return np.exp(
        -(
            np.pi**2 * w_rad**2 * u_hz**2
        ) / (4.0 * np.log(2.0))
    )


def visibility_model(u_hz, Reff_rad, F0, w_uas):
    """
    Modelo efectivo de anillo fino con envolvente gaussiana:
        V(u) = F0 * |J0(2π R_eff u)| * E(u)
    La comparación observacional usa |V|.
    """
    bessel_term = sp.j0(2.0 * np.pi * Reff_rad * u_hz)
    envelope = gaussian_envelope(u_hz, w_uas)
    return F0 * np.abs(bessel_term) * envelope


# ==============================================================================
# 5. AJUSTE DE PARÁMETROS NUISANCE (F0, w) PARA R_eff FIJO
# ==============================================================================

def fit_nuisance_params(u_hz, y, sigma, Reff_rad):
    """
    Ajusta F0 y w para un R_eff fijo usando least_squares.
    Retorna dict con F0, w_uas, model, chi2, chi2_reduced, dof.
    """

    def residuals(params):
        F0, w_uas = params
        model = visibility_model(u_hz, Reff_rad, F0, w_uas)
        return (y - model) / sigma

    # Valores iniciales razonables
    p0 = [np.percentile(y, 95), 12.0]

    # Límites físicos
    bounds = ([0.0, 0.1], [100.0, 100.0])

    result = least_squares(residuals, p0, bounds=bounds)

    F0_best, w_best = result.x
    model_best = visibility_model(u_hz, Reff_rad, F0_best, w_best)

    residual = (y - model_best) / sigma
    chi2 = np.sum(residual**2)
    n = len(y)
    k = 2  # F0, w
    dof = max(n - k, 1)
    chi2_reduced = chi2 / dof

    return {
        "F0": F0_best,
        "w_uas": w_best,
        "model": model_best,
        "chi2": chi2,
        "chi2_reduced": chi2_reduced,
        "dof": dof,
        "success": result.success
    }


# ==============================================================================
# 6. FUNCIÓN PRINCIPAL DE ANÁLISIS
# ==============================================================================

def main():
    print("=" * 72)
    print("TESIS UNMSM — ANÁLISIS EHT Sgr A* 2024-D02-01")
    print("Schwarzschild vs Kerr-Newman (carga débil δ)")
    print("=" * 72)

    # 1. Cargar datos
    u_raw, v_raw, vis_complex, sigma_raw, obs = cargar_datos_eht()

    # 2. Preprocesar
    u_glambda, vis_amp, sigma_raw = preprocesar_visibilidades(
        u_raw, v_raw, vis_complex, sigma_raw
    )

    # 3. Convertir a Hz
    u_data_hz = u_glambda * 1e9

    # 4. Calcular radios efectivos usando eht_tesis.modelo
    # Schwarzschild (δ = 0)
    Reff_schwarz_uas = Reff_angular(0.0, THETA_G_SGRA)
    Reff_schwarz_rad = Reff_schwarz_uas * UAS

    # Kerr-Newman (δ = 0.1)
    Reff_kn_uas = Reff_angular(DELTA, THETA_G_SGRA)
    Reff_kn_rad = Reff_kn_uas * UAS

    factor_kn = ratio_Reff(DELTA)

    # 5. Información geométrica
    print("\n" + "=" * 72)
    print("                 GEOMETRÍA DEL MODELO")
    print("=" * 72)
    print(f"θ_g (Sgr A*)                = {THETA_G_SGRA:.4f} μas")
    print(f"R_eff Schwarzschild (δ=0)   = {Reff_schwarz_uas:.6f} μas")
    print(f"δ                           = {DELTA:.4f}")
    print(f"Factor KN ratio_Reff(δ)     = {factor_kn:.10f}")
    print(f"R_eff Kerr-Newman           = {Reff_kn_uas:.6f} μas")
    print(f"Reducción radial KN         = {100.0 * (1.0 - factor_kn):.6f} %")
    print(f"Diámetro Schwarzschild      = {2.0 * Reff_schwarz_uas:.6f} μas")
    print(f"Diámetro Kerr-Newman        = {2.0 * Reff_kn_uas:.6f} μas")
    print("=" * 72)

    # 6. Ajustar modelo Schwarzschild
    print("\n--> Ajustando modelo Schwarzschild (δ=0)...")
    fit_schwarz = fit_nuisance_params(
        u_data_hz, vis_amp, sigma_raw, Reff_schwarz_rad
    )

    # 7. Ajustar modelo Kerr-Newman
    print("--> Ajustando modelo Kerr-Newman (δ=0.1)...")
    fit_kn = fit_nuisance_params(
        u_data_hz, vis_amp, sigma_raw, Reff_kn_rad
    )

    # 8. Resultados estadísticos
    delta_chi2 = fit_kn["chi2"] - fit_schwarz["chi2"]
    delta_chi2_red = fit_kn["chi2_reduced"] - fit_schwarz["chi2_reduced"]

    print("\n" + "=" * 72)
    print("             RESULTADOS DEL CONTRASTE EHT")
    print("=" * 72)
    print(f"Número de visibilidades       : {len(vis_amp):,}")

    print("\n--- SCHWARZSCHILD (δ=0) ---")
    print(f"F0                            : {fit_schwarz['F0']:.6f} Jy")
    print(f"w                             : {fit_schwarz['w_uas']:.6f} μas")
    print(f"χ²                            : {fit_schwarz['chi2']:.6f}")
    print(f"χ² reducido                   : {fit_schwarz['chi2_reduced']:.6f}")
    print(f"Grados de libertad            : {fit_schwarz['dof']}")

    print("\n--- KERR-NEWMAN (δ=0.1) ---")
    print(f"F0                            : {fit_kn['F0']:.6f} Jy")
    print(f"w                             : {fit_kn['w_uas']:.6f} μas")
    print(f"χ²                            : {fit_kn['chi2']:.6f}")
    print(f"χ² reducido                   : {fit_kn['chi2_reduced']:.6f}")
    print(f"Grados de libertad            : {fit_kn['dof']}")

    print("\n--- COMPARACIÓN ---")
    print(f"Δχ² = χ²_KN - χ²_Sch         : {delta_chi2:+.6f}")
    print(f"Δχ²_reducido                  : {delta_chi2_red:+.6f}")

    if fit_kn["chi2_reduced"] < fit_schwarz["chi2_reduced"]:
        print("\n>>> RESULTADO: El modelo Kerr-Newman presenta un MENOR χ² reducido.")
        print(">>> La carga débil MEJORA el ajuste (reduce error sistemático).")
    else:
        print("\n>>> RESULTADO: El modelo Schwarzschild presenta un MENOR χ² reducido.")
        print(">>> La carga débil NO mejora el ajuste para este dataset.")

    print("=" * 72)

    # 9. Nulos teóricos usando eht_tesis.modelo
    nulls_schwarz = nulos_J0(2 * Reff_schwarz_uas, 0.0, m=1)
    nulls_kn = nulos_J0(2 * Reff_kn_uas, DELTA, m=1)
    null1_schwarz = nulls_schwarz[0]
    null1_kn = nulls_kn[0]

    print("\n--- NULOS TEÓRICOS ---")
    print(f"u₁ Schwarzschild = {null1_schwarz:.6f} Gλ")
    print(f"u₁ Kerr-Newman    = {null1_kn:.6f} Gλ")
    print(f"Desplazamiento    = {null1_kn - null1_schwarz:+.6f} Gλ")
    print(f"Desplazamiento %  = {100.0 * (null1_kn / null1_schwarz - 1.0):+.6f} %")

    # 10. Malla teórica para gráfica
    u_grid_glambda = np.linspace(0.01, 10.0, 2000)
    u_grid_hz = u_grid_glambda * 1e9

    V_schwarz_grid = visibility_model(
        u_grid_hz, Reff_schwarz_rad, fit_schwarz["F0"], fit_schwarz["w_uas"]
    )
    V_kn_grid = visibility_model(
        u_grid_hz, Reff_kn_rad, fit_kn["F0"], fit_kn["w_uas"]
    )

    # 11. GRÁFICA FINAL
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Datos EHT
    ax.errorbar(
        u_glambda, vis_amp, yerr=sigma_raw,
        fmt='o', markersize=2.0, alpha=0.20,
        elinewidth=0.5, capsize=0,
        label=f'EHT Sgr A* 2024-D02-01 ({len(vis_amp):,} visibilidades)'
    )

    # Modelo Schwarzschild
    ax.plot(
        u_grid_glambda, V_schwarz_grid,
        linewidth=2.4,
        label=(
            'Schwarzschild '
            rf'($R_{{eff}}={Reff_schwarz_uas:.2f}\,\mu as$'
            rf', $\chi^2_\nu={fit_schwarz["chi2_reduced"]:.2f}$)'
        )
    )

    # Modelo Kerr-Newman
    ax.plot(
        u_grid_glambda, V_kn_grid,
        '--', linewidth=2.4,
        label=(
            rf'Kerr-Newman '
            rf'($\delta={DELTA}$, '
            rf'$R_{{eff}}={Reff_kn_uas:.2f}\,\mu as$'
            rf', $\chi^2_\nu={fit_kn["chi2_reduced"]:.2f}$)'
        )
    )

    # Primer nulo Schwarzschild
    ax.axvline(null1_schwarz, linestyle=':', linewidth=1.2, alpha=0.7)
    ax.annotate(
        rf'Primer nulo Schwarzschild'
        '\n'
        rf'${null1_schwarz:.2f}\,G\lambda$',
        xy=(null1_schwarz, 0.05),
        xytext=(null1_schwarz + 0.3, 0.25),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9)
    )

    # Primer nulo Kerr-Newman
    ax.axvline(null1_kn, linestyle=':', linewidth=1.2, alpha=0.7)
    ax.annotate(
        rf'Primer nulo Kerr-Newman'
        '\n'
        rf'${null1_kn:.2f}\,G\lambda$',
        xy=(null1_kn, 0.05),
        xytext=(null1_kn + 0.3, 0.40),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9)
    )

    # Formato
    ax.set_title(
        "Sgr A*: visibilidades EHT vs modelo efectivo "
        "Schwarzschild–Kerr-Newman",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel(
        r"Línea de base escalar $u=\sqrt{u^2+v^2}$ [$G\lambda$]",
        fontsize=12
    )
    ax.set_ylabel(
        r"Amplitud de visibilidad $|V(u)|$ [Jy]",
        fontsize=12
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig("comparacion_schwarzschild_vs_kerr_newman.png", dpi=300, bbox_inches='tight')
    print("\n--> Gráfica guardada: comparacion_schwarzschild_vs_kerr_newman.png")
    plt.show()

    # 12. Resumen físico final
    print("\n" + "=" * 72)
    print("                    RESUMEN FÍSICO")
    print("=" * 72)
    print(f"""
1. La carga débil δ = {DELTA} reduce el radio efectivo:

   R_eff(Schwarzschild) = {Reff_schwarz_uas:.4f} μas
   R_eff(Kerr-Newman)   = {Reff_kn_uas:.4f} μas

2. Como u₁ ∝ 1/R_eff:

   u₁(Schwarzschild) = {null1_schwarz:.4f} Gλ
   u₁(Kerr-Newman)   = {null1_kn:.4f} Gλ

3. Por tanto, la carga desplaza el primer nulo hacia
   líneas de base mayores.

4. El contraste estadístico se realiza utilizando
   exactamente el mismo dataset EHT y ajustando
   los parámetros de emisión F0 y w para ambos modelos.

5. Este resultado corresponde al modelo efectivo radial
   de anillo fino de la tesis.

6. No debe interpretarse todavía como un ajuste completo
   de la sombra 2D de Kerr-Newman.
""")
    print("=" * 72)

    return fit_schwarz, fit_kn, null1_schwarz, null1_kn


if __name__ == "__main__":
    fit_schwarz, fit_kn, null1_schwarz, null1_kn = main()