"""
Módulo de ajuste: modelo de visibilidad J0 + gaussiano + chi2 con sistemático y prior.
"""
import numpy as np
from scipy.special import j0
from scipy.optimize import least_squares
from eht_tesis.modelo import UAS, ratio_Reff, validar_delta


def visibilidad_modelo(uv_Gl, F, theta_g_uas, sigma_b_uas, delta):
    """
    Modelo de visibilidad |V(u)| para anillo con desenfoque gaussiano.

    Parámetros:
    - uv_Gl: array de uv distance en Gλ
    - F: flujo total (Jy)
    - theta_g_uas: tamaño angular gravitacional θ_g = GM/(c²D) en µas
    - sigma_b_uas: desenfoque gaussiano σ_b en µas (ancho del anillo)
    - delta: parámetro de carga

    Retorna:
    - |V| en Jy
    """
    validar_delta(delta)
    # Radio angular efectivo: R_eff = √27 * θ_g * ratio_Reff(δ)  [µas]
    R_eff_uas = np.sqrt(27.0) * theta_g_uas * ratio_Reff(delta)
    # Convertir a radianes
    R_eff_rad = R_eff_uas * UAS
    # u en λ (no Gλ)
    u_lambda = uv_Gl * 1e9
    # Modelo: J0(2π R u) * exp(-2π² σ_b² u²)
    V = F * np.abs(j0(2.0 * np.pi * R_eff_rad * u_lambda))
    if sigma_b_uas > 0:
        sigma_b_rad = sigma_b_uas * UAS
        V *= np.exp(-2.0 * np.pi**2 * sigma_b_rad**2 * u_lambda**2)
    return V


def residuos(params, uv_Gl, amp_obs, sigma_obs, f_sys=0.10, theta_g_prior=None, theta_g_prior_sigma=None, delta=0.0):
    """
    Residuos normalizados para least_squares.

    params = [logF, log_theta_g, log_sigma_b]  (delta es fijo, pasado por kwarg)
    """
    logF, log_theta_g, log_sigma_b = params

    F = np.exp(logF)
    theta_g = np.exp(log_theta_g)
    sigma_b = np.exp(log_sigma_b)

    validar_delta(delta)

    V_mod = visibilidad_modelo(uv_Gl, F, theta_g, sigma_b, delta)

    # Error efectivo: térmico + sistemático fraccional
    sigma_eff = np.sqrt(sigma_obs**2 + (f_sys * V_mod)**2)

    res = (amp_obs - V_mod) / sigma_eff

    # Prior gaussiano sobre theta_g
    if theta_g_prior is not None and theta_g_prior_sigma is not None:
        prior_res = (theta_g - theta_g_prior) / theta_g_prior_sigma
        res = np.append(res, prior_res)

    return res


def chi2_total(params, uv_Gl, amp_obs, sigma_obs, f_sys=0.10, theta_g_prior=None, theta_g_prior_sigma=None, delta=0.0):
    """Chi-cuadrado total (incluye prior)."""
    res = residuos(params, uv_Gl, amp_obs, sigma_obs, f_sys, theta_g_prior, theta_g_prior_sigma, delta)
    return np.sum(res**2)


def ajustar_delta_fijo(delta, uv_Gl, amp_obs, sigma_obs,
                       f_sys=0.10, theta_g_prior=None, theta_g_prior_sigma=None,
                       x0=None, bounds=None):
    """
    Ajusta F, theta_g, sigma_b para un delta fijo.
    Retorna: (params_opt, chi2, success, message)
    params_opt = [F, theta_g, sigma_b, delta]
    """
    validar_delta(delta)

    # Parámetros iniciales en log-espacio (solo 3: F, theta_g, sigma_b)
    if x0 is None:
        F0 = amp_obs.max() * 1.2
        theta_g0 = 3.8  # µas
        sigma_b0 = 8.0  # µas
        x0 = [np.log(F0), np.log(theta_g0), np.log(sigma_b0)]
    else:
        x0 = list(x0[:3])

    # Límites para F, theta_g, sigma_b (3 parámetros)
    if bounds is None:
        lb = [np.log(0.1), np.log(0.5), np.log(0.1)]
        ub = [np.log(10.0), np.log(10.0), np.log(50.0)]
        bounds = (lb, ub)
    else:
        bounds = list(bounds)
        bounds[0] = bounds[0][:3]
        bounds[1] = bounds[1][:3]

    def fun(p):
        return residuos(p, uv_Gl, amp_obs, sigma_obs, f_sys, theta_g_prior, theta_g_prior_sigma, delta)

    try:
        result = least_squares(fun, x0, bounds=bounds, verbose=0, ftol=1e-8, gtol=1e-8, max_nfev=200)
        p_opt = result.x
        F_opt = np.exp(p_opt[0])
        theta_g_opt = np.exp(p_opt[1])
        sigma_b_opt = np.exp(p_opt[2])
        chi2 = np.sum(result.fun**2)
        n_params = 3 + (1 if theta_g_prior is not None else 0)  # F, theta_g, sigma_b + prior
        n_data = len(amp_obs) + (1 if theta_g_prior is not None else 0)
        chi2_red = chi2 / (n_data - n_params) if n_data > n_params else np.inf
        return [F_opt, theta_g_opt, sigma_b_opt, delta], chi2, chi2_red, result.success, result.message
    except Exception as e:
        print(f'  Error en least_squares (δ={delta}): {e}')
        return None, np.inf, np.inf, False, str(e)


def escanear_delta(deltas, uv_Gl, amp_obs, sigma_obs, **kwargs):
    """
    Escanea delta y reajusta los otros parámetros.
    Retorna arrays: deltas, chi2, chi2_red, F, theta_g, sigma_b, success
    """
    results = []
    for d in deltas:
        params, chi2, chi2_red, success, msg = ajustar_delta_fijo(
            d, uv_Gl, amp_obs, sigma_obs, **kwargs)
        if params is not None:
            results.append((d, params[0], params[1], params[2], chi2, chi2_red, success))
        else:
            results.append((d, np.nan, np.nan, np.nan, np.inf, np.inf, False))

    deltas_arr = np.array([r[0] for r in results])
    F_arr = np.array([r[1] for r in results])
    theta_g_arr = np.array([r[2] for r in results])
    sigma_b_arr = np.array([r[3] for r in results])
    chi2_arr = np.array([r[4] for r in results])
    chi2_red_arr = np.array([r[5] for r in results])
    success_arr = np.array([r[6] for r in results])

    return deltas_arr, F_arr, theta_g_arr, sigma_b_arr, chi2_arr, chi2_red_arr, success_arr


def generar_datos_sinteticos(F_true=1.1, theta_g_true=3.9, sigma_b_true=8.0, delta_true=0.0,
                            uv_Gl=None, sigma_rel=0.05, seed=1):
    """Genera datos sintéticos para tests."""
    rng = np.random.default_rng(seed)
    if uv_Gl is None:
        uv_Gl = np.linspace(0.1, 6.7, 200)
    V_true = visibilidad_modelo(uv_Gl, F_true, theta_g_true, sigma_b_true, delta_true)
    sigma = V_true * sigma_rel
    amp_obs = V_true + rng.normal(0, sigma)
    return uv_Gl, amp_obs, sigma


def ajustar_sin_prior_theta_g(deltas, uv_Gl, amp_obs, sigma_obs, f_sys=0.10, **kwargs):
    """
    Ajuste SIN prior en theta_g — muestra degeneración delta-theta_g.
    """
    return escanear_delta(deltas, uv_Gl, amp_obs, sigma_obs, f_sys=f_sys,
                         theta_g_prior=None, theta_g_prior_sigma=None, **kwargs)