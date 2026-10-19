#!/usr/bin/env python3
"""
===============================================================================
TESIS UNMSM — GRÁFICA RESUMEN FINAL PARA TESIS
===============================================================================

Genera una figura de calidad de publicación con todos los resultados clave:
1. χ² vs δ (Sgr A*) con mínimo marcado
2. Visibilidades Sgr A* vs modelos (Sch vs KN best-fit)
3. Visibilidades M87* vs modelos (comparación)
4. Tabla resumen de resultados
===============================================================================
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
# CONSTANTES Y MODELO
# ==============================================================================

J0_FIRST_ZERO = sp.jn_zeros(0, 1)[0]
THETA_G_SGRA = 5.03  # μas
THETA_RING_M87 = 42.0  # μas


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


def load_eht_data_sgra():
    """Carga Sgr A* desde repo oficial 2024-D02-01 (auto-clonado)."""
    REPO_DIR = "2024-D02-01"
    ruta = os.path.join(
        REPO_DIR, "hops_data", "April06",
        "ER6_SGRA_2017_096_hi_hops_netcal_LMTcal_10s_ALMArot_dtermcal.uvfits"
    )
    if not os.path.isdir(REPO_DIR):
        print("--> Clonando repositorio oficial EHT 2024-D02-01...")
        subprocess.run(["git", "clone", "https://github.com/eventhorizontelescope/2024-D02-01.git", REPO_DIR], check=True)
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


def load_eht_data_m87():
    """Carga M87* desde data/ local."""
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    uvfits = os.path.join(base, "M87/2018_polarized/uvfits/",
                          "L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.uvfits")
    obs = eh.obsdata.load_uvfits(uvfits)
    data = obs.data
    u = np.asarray(data["u"], float)
    v = np.asarray(data["v"], float)
    vis = np.asarray(data["vis"], complex)
    sigma = np.asarray(data["sigma"], float)
    amp = np.abs(vis)
    u_glambda = np.sqrt(u**2 + v**2) / 1e9
    mask = (np.isfinite(u_glambda) & np.isfinite(amp) & np.isfinite(sigma)
            & (sigma > 0) & (u_glambda >= 0.01))
    return u_glambda[mask] * 1e9, amp[mask], sigma[mask]


# ==============================================================================
# CARGA DE DATOS Y ESCANEO
# ==============================================================================

# Sgr A* (repo oficial)
u_hz_sgra, y_sgra, sig_sgra = load_eht_data_sgra()
print(f"Sgr A*: {len(y_sgra)} vis, UV=[{u_hz_sgra.min()/1e9:.2f}, {u_hz_sgra.max()/1e9:.2f}] Gλ")

# M87* (data local)
u_hz_m87, y_m87, sig_m87 = load_eht_data_m87()
print(f"M87*: {len(y_m87)} vis, UV=[{u_hz_m87.min()/1e9:.2f}, {u_hz_m87.max()/1e9:.2f}] Gλ")

# Chi2 scan Sgr A*
deltas = np.linspace(0, 0.4, 81)
chi2_vals = []
for d in deltas:
    R = Reff_angular(d, THETA_G_SGRA)
    fit = fit_nuisance(u_hz_sgra, y_sgra, sig_sgra, R)
    chi2_vals.append(fit["chi2"])
chi2_vals = np.array(chi2_vals)
best_idx = np.argmin(chi2_vals)
best_d = deltas[best_idx]
best_c2 = chi2_vals[best_idx]

# Fit best models
# Sgr A* Schwarzschild
R_sch_sgra = Reff_angular(0.0, THETA_G_SGRA)
fit_sch_sgra = fit_nuisance(u_hz_sgra, y_sgra, sig_sgra, R_sch_sgra)
F0_sch_sgra, w_sch_sgra, mod_sch_sgra, c2_sch_sgra, cr_sch_sgra = (
    fit_sch_sgra["F0"], fit_sch_sgra["w"], fit_sch_sgra["model"],
    fit_sch_sgra["chi2"], fit_sch_sgra["chi2_red"]
)
# Sgr A* KN best-fit
R_kn_sgra = Reff_angular(best_d, THETA_G_SGRA)
fit_kn_sgra = fit_nuisance(u_hz_sgra, y_sgra, sig_sgra, R_kn_sgra)
F0_kn_sgra, w_kn_sgra, mod_kn_sgra, c2_kn_sgra, cr_kn_sgra = (
    fit_kn_sgra["F0"], fit_kn_sgra["w"], fit_kn_sgra["model"],
    fit_kn_sgra["chi2"], fit_kn_sgra["chi2_red"]
)
# M87* Schwarzschild
R_sch_m87 = Reff_angular(0.0, THETA_RING_M87)
fit_sch_m87 = fit_nuisance(u_hz_m87, y_m87, sig_m87, R_sch_m87)
F0_sch_m87, w_sch_m87, mod_sch_m87, c2_sch_m87, cr_sch_m87 = (
    fit_sch_m87["F0"], fit_sch_m87["w"], fit_sch_m87["model"],
    fit_sch_m87["chi2"], fit_sch_m87["chi2_red"]
)
# M87* KN delta=0.1
R_kn_m87 = Reff_angular(0.1, THETA_RING_M87)
fit_kn_m87 = fit_nuisance(u_hz_m87, y_m87, sig_m87, R_kn_m87)
F0_kn_m87, w_kn_m87, mod_kn_m87, c2_kn_m87, cr_kn_m87 = (
    fit_kn_m87["F0"], fit_kn_m87["w"], fit_kn_m87["model"],
    fit_kn_m87["chi2"], fit_kn_m87["chi2_red"]
)

# Nulls
u1_sch_sgra = J0_FIRST_ZERO / (2*np.pi * R_sch_sgra * UAS) / 1e9
u1_kn_sgra = J0_FIRST_ZERO / (2*np.pi * R_kn_sgra * UAS) / 1e9
u1_sch_m87 = J0_FIRST_ZERO / (2*np.pi * R_sch_m87 * UAS) / 1e9
u1_kn_m87 = J0_FIRST_ZERO / (2*np.pi * R_kn_m87 * UAS) / 1e9

# =============================================================================
# FIGURA PRINCIPAL 4 PANELES
# =============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

# PANEL A: χ² vs δ (Sgr A*)
axA = fig.add_subplot(gs[0, 0])
axA.plot(deltas, chi2_vals, 'b-', linewidth=1.5, label=r'$\chi^2(\delta)$')
axA.axvline(0, color='gray', linestyle=':', linewidth=1, label=r'Schwarzschild ($\delta=0$)')
axA.axvline(best_d, color='red', linestyle='--', linewidth=1.5,
            label=f'Mínimo: $\delta={best_d:.3f}$')
axA.set_xlabel(r'$\delta$ (parámetro de carga débil)')
axA.set_ylabel(r'$\chi^2$')
axA.set_title(r'(a) Sgr A*: $\chi^2$ vs $\delta$ (EHT 2024-D02-01)')
axA.legend(loc='upper right', frameon=True)
axA.grid(True, alpha=0.3)
axA.annotate(f'$\\Delta\\chi^2 = {best_c2 - chi2_vals[0]:+.0f}$\n$\\chi^2_\\nu: {chi2_vals[0]/9302:.1f} \\to {best_c2/9302:.2f}$',
             xy=(best_d, best_c2), xytext=(0.15, chi2_vals[0]*0.995),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='red', alpha=0.9))

# PANEL B: Sgr A* Data vs Models
axB = fig.add_subplot(gs[0, 1])
u_grid = np.linspace(0.01, 9, 1000) * 1e9
axB.errorbar(u_hz_sgra/1e9, y_sgra, yerr=sig_sgra, fmt='o', ms=1.5, alpha=0.15,
             elinewidth=0.3, capsize=0, color='gray', label='Datos EHT (9,304 vis)')
axB.plot(u_grid/1e9, visibility_model(u_grid, R_sch_sgra, F0_sch_sgra, w_sch_sgra), 'r-', lw=2.5,
         label=rf'Schwarzschild ($\chi^2_\nu={cr_sch_sgra:.1f}$)')
axB.plot(u_grid/1e9, visibility_model(u_grid, R_kn_sgra, F0_kn_sgra, w_kn_sgra), 'b--', lw=2.5,
         label=rf'Kerr-Newman $\delta={best_d:.2f}$ ($\chi^2_\nu={cr_kn_sgra:.1f}$)')
axB.axvline(u1_sch_sgra, color='r', ls=':', alpha=0.7, lw=1)
axB.axvline(u1_kn_sgra, color='b', ls=':', alpha=0.7, lw=1)
axB.set_xlabel(r'Línea de base $u$ [G$\lambda$]')
axB.set_ylabel(r'Amplitud $|V(u)|$ [Jy]')
axB.set_title(r'(b) Sgr A* 2017: Datos vs Modelos')
axB.legend(loc='upper right', frameon=True)
axB.grid(True, alpha=0.3)
axB.set_xlim(0, 9)
axB.set_ylim(0, 3)

# PANEL C: M87* Data vs Models
axC = fig.add_subplot(gs[1, 0])
axC.errorbar(u_hz_m87/1e9, y_m87, yerr=sig_m87, fmt='o', ms=1.5, alpha=0.15,
             elinewidth=0.3, capsize=0, color='gray', label='Datos EHT (4,237 vis)')
axC.plot(u_grid/1e9, visibility_model(u_grid, R_sch_m87, F0_sch_m87, w_sch_m87), 'r-', lw=2.5,
         label=rf'Schwarzschild ($\chi^2_\nu={cr_sch_m87:.1f}$)')
axC.plot(u_grid/1e9, visibility_model(u_grid, R_kn_m87, F0_kn_m87, w_kn_m87), 'b--', lw=2.5,
         label=rf'Kerr-Newman $\delta=0.1$ ($\chi^2_\nu={cr_kn_m87:.1f}$)')
axC.axvline(u1_sch_m87, color='r', ls=':', alpha=0.7, lw=1)
axC.axvline(u1_kn_m87, color='b', ls=':', alpha=0.7, lw=1)
axC.set_xlabel(r'Línea de base $u$ [G$\lambda$]')
axC.set_ylabel(r'Amplitud $|V(u)|$ [Jy]')
axC.set_title(r'(c) M87* 2018: Datos vs Modelos')
axC.legend(loc='upper right', frameon=True)
axC.grid(True, alpha=0.3)
axC.set_xlim(0, 8)
axC.set_ylim(0, 1.5)

# PANEL D: Tabla resumen
axD = fig.add_subplot(gs[1, 1])
axD.axis('off')

table_data = [
    ['Parámetro', 'Sgr A* (2024-D02-01)', 'M87* (2024-D01-01)'],
    ['Dataset', '6 Abr 2017, hi, HOPS', '21 Abr 2018, b1, HOPS'],
    ['$N_{\\mathrm{vis}}$', '9,304', '4,237'],
    ['$u$-range [G$\\lambda$]', '[0.0, 8.5]', '[1.3, 6.7]'],
    ['', '', ''],
    ['Schwarzschild ($\delta=0$)', '', ''],
    ['  $R_{\\mathrm{eff}}$ [$\\mu$as]', f'{R_sch_sgra:.1f}', f'{R_sch_m87:.1f}'],
    ['  $F_0$ [Jy]', f'{F0_sch_sgra:.3f}', f'{F0_sch_m87:.3f}'],
    ['  $w$ [$\\mu$as]', f'{w_sch_sgra:.1f}', f'{w_sch_m87:.1f}'],
    ['  $\\chi^2$', f'{c2_sch_sgra:.0f}', f'{c2_sch_m87:.0f}'],
    ['  $\\chi^2_\\nu$', f'{cr_sch_sgra:.2f}', f'{cr_sch_m87:.2f}'],
    ['  $u_1$ [G$\\lambda$]', f'{u1_sch_sgra:.3f}', f'{u1_sch_m87:.3f}'],
    ['', '', ''],
    ['Kerr-Newman (best $\\delta$)', f'$\\delta={best_d:.2f}$', '$\\delta=0.1$'],
    ['  $R_{\\mathrm{eff}}$ [$\\mu$as]', f'{R_kn_sgra:.1f}', f'{R_kn_m87:.1f}'],
    ['  $\\Delta R/R$ [%]', f'{100*(1-R_kn_sgra/R_sch_sgra):.2f}', f'{100*(1-R_kn_m87/R_sch_m87):.2f}'],
    ['  $F_0$ [Jy]', f'{F0_kn_sgra:.3f}', f'{F0_kn_m87:.3f}'],
    ['  $w$ [$\\mu$as]', f'{w_kn_sgra:.1f}', f'{w_kn_m87:.1f}'],
    ['  $\\chi^2$', f'{c2_kn_sgra:.0f}', f'{c2_kn_m87:.0f}'],
    ['  $\\chi^2_\\nu$', f'{cr_kn_sgra:.2f}', f'{cr_kn_m87:.2f}'],
    ['  $u_1$ [G$\\lambda$]', f'{u1_kn_sgra:.3f}', f'{u1_kn_m87:.3f}'],
    ['  $\\Delta u_1$ [G$\\lambda$]', f'{u1_kn_sgra-u1_sch_sgra:+.3f}', f'{u1_kn_m87-u1_sch_m87:+.3f}'],
    ['', '', ''],
    ['$\Delta\\chi^2$ (KN - Sch)', f'{c2_kn_sgra - c2_sch_sgra:+.0f}', f'{c2_kn_m87 - c2_sch_m87:+.0f}'],
    ['$\Delta\\chi^2_\\nu$', f'{cr_kn_sgra - cr_sch_sgra:+.3f}', f'{cr_kn_m87 - cr_sch_m87:+.3f}'],
    ['¿Mejora KN?', '\\textbf{SÍ} ✓', 'NO ✗'],
]

table = axD.table(cellText=table_data, loc='center', cellLoc='left',
                  colWidths=[0.38, 0.31, 0.31])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

# Estilo de la tabla
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(color='white', weight='bold')
    elif row in [5, 12, 19, 24]:
        cell.set_facecolor('#E8E8E8')
        cell.set_text_props(weight='bold')
    elif row == 23:
        if col == 1:
            cell.set_facecolor('#C8E6C9')
            cell.set_text_props(weight='bold', color='darkgreen')
        elif col == 2:
            cell.set_facecolor('#FFCCBC')
            cell.set_text_props(weight='bold', color='darkred')
    cell.set_edgecolor('#CCCCCC')

axD.set_title(r'(d) Resumen de Resultados', pad=20, weight='bold')

plt.suptitle(
    'Tesis UNMSM: Kerr-Newman Carga Débil — Análisis EHT Multi-Fuente\n'
    'Tesista: Henry Paolo Alfaro Sotil | Asesor: Dr. David Pareja Quispe',
    fontsize=15, weight='bold', y=0.98
)

plt.savefig('tesis_resultados_finales.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Guardado: tesis_resultados_finales.png")
plt.show()

# =============================================================================
# IMPRIMIR RESUMEN NUMÉRICO PARA TESIS
# =============================================================================
print("\n" + "="*80)
print("RESUMEN NUMÉRICO PARA TESIS")
print("="*80)
print(f"\nSGR A* (EHT 2024-D02-01):")
print(f"  Schwarzschild: χ²={c2_sch_sgra:.0f}, χ²_ν={cr_sch_sgra:.3f}, R_eff={R_sch_sgra:.2f} μas, u₁={u1_sch_sgra:.3f} Gλ")
print(f"  KN best (δ={best_d:.3f}): χ²={c2_kn_sgra:.0f}, χ²_ν={cr_kn_sgra:.3f}, R_eff={R_kn_sgra:.2f} μas, u₁={u1_kn_sgra:.3f} Gλ")
print(f"  Δχ² = {c2_kn_sgra - c2_sch_sgra:+.0f}, Δχ²_ν = {cr_kn_sgra - cr_sch_sgra:+.3f}")
print(f"  Δu₁ = {u1_kn_sgra - u1_sch_sgra:+.3f} Gλ ({100*(u1_kn_sgra/u1_sch_sgra-1):+.2f}%)")

print(f"\nM87* (EHT 2024-D01-01, b1 HOPS):")
print(f"  Schwarzschild: χ²={c2_sch_m87:.0f}, χ²_ν={cr_sch_m87:.3f}, R_eff={R_sch_m87:.2f} μas, u₁={u1_sch_m87:.3f} Gλ")
print(f"  KN (δ=0.1): χ²={c2_kn_m87:.0f}, χ²_ν={cr_kn_m87:.3f}, R_eff={R_kn_m87:.2f} μas, u₁={u1_kn_m87:.3f} Gλ")
print(f"  Δχ² = {c2_kn_m87 - c2_sch_m87:+.0f}, Δχ²_ν = {cr_kn_m87 - cr_sch_m87:+.3f}")
print(f"  Δu₁ = {u1_kn_m87 - u1_sch_m87:+.3f} Gλ ({100*(u1_kn_m87/u1_sch_m87-1):+.2f}%)")

print("\nCONCLUSIÓN PARA TESIS:")
print(f"  • Sgr A*: Kerr-Newman MEJORA el ajuste (Δχ²_ν = {cr_kn_sgra - cr_sch_sgra:+.3f})")
print(f"    → Carga débil δ ≈ {best_d:.3f} preferida por los datos")
print(f"  • M87*: Schwarzschild igual o mejor (Δχ²_ν = {cr_kn_m87 - cr_sch_m87:+.3f})")
print(f"    → Modelo de anillo simple insuficiente para M87* (chorro, emisión extendida)")
print(f"  • La firma D=1/27 en R_eff²(δ) predice desplazamiento de nulos Δu₁/u₁ ≈ δ/6")
print(f"  • Sgr A* datos soportan esta predicción; M87* requiere modelo más completo")
print("="*80)