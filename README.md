# Análisis Perturbativo de la Esfera de Fotones y Detectabilidad de la Sombra en el Espaciotiempo de Kerr-Newman con Carga Débil

**Tesis de pregrado** — Universidad Nacional Mayor de San Marcos (UNMSM)  
**Tesista:** Henry Paolo Alfaro Sotil  
**Asesor:** Dr. David Pareja Quispe  
**Facultad:** Ciencias Físicas  

---

## Resumen Científico

Este repositorio presenta el análisis completo para validar un modelo de **agujero negro Kerr-Newman con carga débil** (δ ≪ 1) contra datos interferométricos reales del **Event Horizon Telescope (EHT)**.

### Metodología

1. **Geometría exacta de la esfera de fotones**: Radio Kerr-Newman sin expansión en Taylor  
   r₀(δ) = 1.5(1 + √(1 - 8δ/9)) r_g

2. **Radio efectivo de la sombra**: R_eff(δ) = √27 θ_g · ratio_Reff(δ) con coeficientes verificados simbólicamente

3. **Modelo de visibilidad**: Anillo fino + envolvente gaussiana  
   |V(u)| = F₀ |J₀(2πR_eff u)| × exp(-π²w²u²/4ln2)

4. **Inferencia estadística**: χ² en amplitudes de visibilidad con incertidumbre sistemática del 10%

5. **Datos oficiales EHT**:
   - **Sgr A***: Campaña 2017 (2024-D02-01) — 9,304 visibilidades, banda alta, calibración HOPS+LMT
   - **M87***: Campaña 2018 (2024-D01-01) — 4,237 visibilidades, banda b1, calibración HOPS

---

## Resultados Principales

| Fuente | Modelo | δ | R_eff [μas] | χ²_ν | u₁ [Gλ] | Δχ²_ν |
|--------|--------|---|-------------|------|---------|-------|
| **Sgr A*** | Schwarzschild | 0.000 | 26.14 | 123.28 | 3.021 | — |
| **Sgr A*** | **Kerr-Newman (best-fit)** | **0.290** | **24.79** | **118.96** | **3.184** | **-4.32** ✓ |
| **M87*** | Schwarzschild | 0.000 | 21.00 | 67.21 | 3.759 | — |
| **M87*** | Kerr-Newman | 0.100 | 20.64 | 77.95 | 3.824 | +10.74 ✗ |

> **Interpretación física**: Sgr A* muestra una mejora estadística con δ ≈ 0.29 (Δχ²_ν = -4.32), consistente con el desplazamiento teórico de nulos Δu₁/u₁ ≈ δ/6. Sin embargo, la señal relativa (ΔR/R ≈ 5%) permanece por debajo de las incertidumbres sistemáticas dominantes (θ_g ~12%). M87* requiere modelado de estructura de chorro y emisión extendida no capturada por el modelo de anillo simple.

---

## Figura Principal

![Resultados finales](docs/FIGURAS_TESIS/tesis_resultados_finales.png)

**Figura 1.** Resumen cuádruple: (a) Escaneo χ² vs δ para Sgr A* con mínimo en δ = 0.290, (b) Visibilidades Sgr A* vs modelos Schwarzschild y Kerr-Newman best-fit, (c) Visibilidades M87* vs modelos, (d) Tabla comparativa de parámetros ajustados.

---

## Estructura del Repositorio

```
tesis-kerr-newman-eht/
├── src/
│   └── eht_tesis/                 # Paquete Python principal
│       ├── __init__.py
│       ├── modelo.py              # Geometría exacta: ratio_Reff, r0_foton, nulos_J0
│       ├── ajuste.py              # Modelo de visibilidad, residuos, χ², escaneo δ
│       └── detectabilidad.py      # Análisis de sensibilidad y límites de detección
├── scripts/
│   ├── analyze_sgra.py            # Análisis principal Sgr A* (repo 2024-D02-01 auto-clonado)
│   ├── scan_delta_chi2.py         # Escaneo χ² vs δ → best-fit δ ≈ 0.29
│   ├── validate_multi_source.py   # Validación cruzada M87* + Sgr A* (data/ local)
│   └── plot_final_summary.py      # Figura final 4 paneles (calidad publicación)
├── data/
│   ├── M87/2018_polarized/uvfits/     # UVFITS M87* 2018 b1 HOPS
│   └── SgrA/2017_first_image/uvfits_besttime/  # UVFITS Sgr A* 2017 besttime
├── docs/
│   ├── INFORME_TESIS_DETALLADO.pdf    # Informe técnico completo (19 págs)
│   ├── INFORME_TESIS_DETALLADO.tex    # Fuente LaTeX compilable
│   ├── INFORME_TESIS_CODIGO_CORREGIDO.md
│   └── FIGURAS_TESIS/
│       └── tesis_resultados_finales.png  # Figura principal (300 DPI)
├── notebooks/
│   └── pipeline_tesis.ipynb       # Pipeline interactivo completo
├── tests/
│   └── test_modelo.py             # Verificación simbólica de coeficientes
├── pyproject.toml                 # Configuración paquete Python
├── requirements.txt               # Dependencias
├── .python-version                # Python 3.11
├── LICENSE                        # MIT
└── .gitignore
```

> **Nota**: El directorio `2024-D02-01/` (repositorio oficial EHT Sgr A*) se clona automáticamente al ejecutar `analyze_sgra.py` o `scan_delta_chi2.py`. No se incluye en el repositorio por tamaño.

---

## Instalación Rápida

```bash
git clone https://github.com/elbrujo325/tesis-kerr-newman-eht.git
cd tesis-kerr-newman-eht

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Dependencias Clave
- `numpy`, `scipy`, `matplotlib` — Cálculo numérico y visualización
- `astropy` — Requerido por `ehtim`
- `ehtim` — Event Horizon Telescope Imaging Library
- `sympy` — Verificación algebraica simbólica

---

## Pipeline de Análisis

### 1. Análisis Principal Sgr A* (con datos oficiales EHT)
```bash
python scripts/analyze_sgra.py
```
Genera: `comparacion_schwarzschild_vs_kerr_newman.png`

### 2. Escaneo χ² vs δ — Determina best-fit
```bash
python scripts/scan_delta_chi2.py
```
Genera: `chi2_scan_delta.png`, `chi2_scan_results.npz`

### 3. Validación Multi-Fuente (M87* + Sgr A*)
```bash
python scripts/validate_multi_source.py
```
Genera: `comparacion_M87_2018.png`, `comparacion_Sgr_A_2017.png`

### 4. Figura Final para Tesis (4 paneles, 300 DPI)
```bash
python scripts/plot_final_summary.py
```
Genera: `tesis_resultados_finales.png`

### 5. Tests de Verificación Teórica
```bash
pytest tests/
```
Verifica coeficientes con `sympy`: 7/216, 13/216, 1/27

---

## Pipeline Interactivo (Jupyter)

Ejecuta todo el análisis paso a paso con visualizaciones intermedias:

```bash
jupyter notebook notebooks/pipeline_tesis.ipynb
```

Incluye: carga de datos, geometría Kerr-Newman, modelo de visibilidad, ajuste de parámetros, escaneo χ², comparación multi-fuente, y generación de figura final — todo en celdas reproducibles.

---

## Datos EHT Incluidos

Los archivos UVFITS oficiales están versionados en `data/`:
- **M87* 2018**: `data/M87/2018_polarized/uvfits/L2V1_M87_2018_111_b1_hops_netcal_10s_StokesI.uvfits`
- **Sgr A* 2017**: `data/SgrA/2017_first_image/uvfits_besttime/ER6_SGRA_2017_097_hi_hops_netcal-LMTcal-besttime_StokesI.uvfits`

---

## Documentación Técnica

| Archivo | Descripción |
|---------|-------------|
| `docs/INFORME_TESIS_DETALLADO.pdf` | **Informe técnico completo** (álgebra, tablas, figuras) |
| `docs/INFORME_TESIS_DETALLADO.tex` | Fuente LaTeX compilable |
| `docs/INFORME_TESIS_CODIGO_CORREGIDO.md` | Resumen de implementación |
| `docs/FIGURAS_TESIS/tesis_resultados_finales.png` | **Figura principal** para tesis (300 DPI) |

---

## Licencia

MIT License — ver [LICENSE](LICENSE).

---

## Contacto

**Henry Paolo Alfaro Sotil**  
UNMSM — Facultad de Ciencias Físicas  
✉️ henry.alfaro1@unmsm.edu.pe