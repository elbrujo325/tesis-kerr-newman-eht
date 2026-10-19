"""
Tests para eht_tesis.modelo
Ejecutar con: pytest -q tests/test_modelo.py
"""
import numpy as np
import pytest
from eht_tesis.modelo import (
    UAS,
    r0_foton,
    ratio_Reff,
    ratio_Reff_taylor,
    R_eff_squared_taylor,
    nulos_J0,
    diametro_para_nulo1,
    u1_u2_ratio,
    u1_taylor_ratio,
    b_critico_squared,
    validar_delta,
)


class TestModeloBasico:
    """Tests de valores exactos en puntos conocidos."""

    def test_UAS_valor(self):
        assert abs(UAS - 4.84813681109536e-12) < 1e-15

    def test_r0_foton_cero(self):
        assert abs(r0_foton(0.0) - 3.0) < 1e-12

    def test_r0_foton_delta_max(self):
        # δ = 9/8 → sqrt(1 - 8δ/9) = 0 → r0 = 1.5
        assert abs(r0_foton(9.0 / 8.0) - 1.5) < 1e-12

    def test_ratio_Reff_cero(self):
        assert abs(ratio_Reff(0.0) - 1.0) < 1e-12

    def test_ratio_Reff_005(self):
        assert abs(ratio_Reff(0.05) - 0.991584) < 1e-6

    def test_ratio_Reff_010(self):
        assert abs(ratio_Reff(0.10) - 0.982996) < 1e-6

    def test_ratio_Reff_015(self):
        assert abs(ratio_Reff(0.15) - 0.974223) < 1e-6

    def test_nulos_J0_theta42(self):
        nulos = nulos_J0(42.0, 0.0, m=2)
        assert abs(nulos[0] - 3.759) < 0.005
        assert abs(nulos[1] - 8.629) < 0.005

    def test_u1_u2_ratio(self):
        assert abs(u1_u2_ratio() - 2.2954) < 1e-3

    def test_diametro_para_nulo1_3p4(self):
        theta = diametro_para_nulo1(3.4, 0.0)
        assert abs(theta - 46.44) < 0.05


class TestSeriesTaylor:
    """Tests de series de Taylor (coeficientes exactos verificados con sympy)."""

    def test_ratio_Reff_taylor_orden1(self):
        # ratio = 1 - δ/6 + O(δ²)
        d = 0.1
        exacto = ratio_Reff(d)
        taylor = ratio_Reff_taylor(d, orden=1)
        assert abs(exacto - taylor) < 1e-3  # error O(δ²) ~ 0.01² = 1e-4

    def test_ratio_Reff_taylor_orden2(self):
        d = 0.01
        exacto = ratio_Reff(d)
        taylor = ratio_Reff_taylor(d, orden=2)
        # error O(δ³) ≈ 151/11664 * 1e-6 ≈ 1.3e-8
        assert abs(exacto - taylor) < 2e-8

    def test_R_eff_squared_taylor_coef_delta2(self):
        """Verifica que el coeficiente de δ² en R_eff²/(27r_g²) es -1/27."""
        # R_eff²/(27r_g²) = ratio² = 1 - δ/3 - δ²/27 - 11δ³/729
        d = 0.01
        exacto = ratio_Reff(d)**2
        taylor2 = R_eff_squared_taylor(d, orden=2)
        assert abs(exacto - taylor2) < 2e-8

    def test_u1_taylor_ratio_coef_delta2(self):
        """Verifica que el coeficiente de δ² en u₁(δ)/u₁(0) es 13/216."""
        d = 0.01
        exacto = 1.0 / ratio_Reff(d)
        taylor = u1_taylor_ratio(d, orden=2)
        assert abs(exacto - taylor) < 3e-8


class TestValidacion:
    def test_delta_negativo_raise(self):
        with pytest.raises(ValueError):
            ratio_Reff(-0.01)

    def test_delta_mayor_9_8_raise(self):
        with pytest.raises(ValueError):
            ratio_Reff(1.2)  # 9/8 = 1.125

    def test_delta_en_borde_9_8_ok(self):
        # Debe funcionar sin error
        ratio_Reff(9.0 / 8.0)


class TestSimbolico:
    """Tests con sympy para verificar coeficientes analíticamente."""

    def test_coeficientes_taylor_simbolicos(self):
        try:
            import sympy as sp
        except ImportError:
            pytest.skip("sympy no instalado")

        delta = sp.symbols('delta', real=True, nonnegative=True)
        r0 = 1.5 * (1 + sp.sqrt(1 - 8*delta/9))
        ratio_sq = 2 * r0**3 / (r0 - 1) / 27
        ratio = sp.sqrt(ratio_sq)

        # Serie de ratio² = R_eff²/(27 r_g²)
        series_ratio_sq = sp.series(ratio_sq, delta, 0, 4).removeO()
        # Esperado: 1 - δ/3 - δ²/27 - 11δ³/729
        assert abs(float(series_ratio_sq.coeff(delta, 0)) - 1.0) < 1e-12
        assert abs(float(series_ratio_sq.coeff(delta, 1)) + 1.0/3.0) < 1e-12
        assert abs(float(series_ratio_sq.coeff(delta, 2)) + 1.0/27.0) < 1e-12
        assert abs(float(series_ratio_sq.coeff(delta, 3)) + 11.0/729.0) < 1e-12

        # Serie de ratio = R_eff/(3√3 r_g)
        series_ratio = sp.series(ratio, delta, 0, 4).removeO()
        # Esperado: 1 - δ/6 - 7δ²/216 - 151δ³/11664
        assert abs(float(series_ratio.coeff(delta, 0)) - 1.0) < 1e-12
        assert abs(float(series_ratio.coeff(delta, 1)) + 1.0/6.0) < 1e-12
        assert abs(float(series_ratio.coeff(delta, 2)) + 7.0/216.0) < 1e-12
        assert abs(float(series_ratio.coeff(delta, 3)) + 151.0/11664.0) < 1e-12

        # Serie de 1/ratio = u₁(δ)/u₁(0)
        series_inv = sp.series(1/ratio, delta, 0, 3).removeO()
        # Esperado: 1 + δ/6 + 13δ²/216
        assert abs(float(series_inv.coeff(delta, 0)) - 1.0) < 1e-12
        assert abs(float(series_inv.coeff(delta, 1)) - 1.0/6.0) < 1e-12
        assert abs(float(series_inv.coeff(delta, 2)) - 13.0/216.0) < 1e-12

    def test_b_critico_en_r0(self):
        """Verifica b²(r₀) = 2 r₀³/(r₀-1) para varios δ."""
        for d in [0.0, 0.05, 0.1, 0.15, 0.5]:
            r0 = r0_foton(d)
            b2 = b_critico_squared(r0, d)
            esperado = 2.0 * r0**3 / (r0 - 1.0)
            assert abs(b2 - esperado) < 1e-10


class TestConsistenciaNumerica:
    """Tests de consistencia entre funciones."""

    def test_nulos_J0_consistente_con_diametro(self):
        """nulos_J0(diametro_para_nulo1(u1), 0)[0] == u1"""
        for u1 in [3.0, 3.4, 3.76, 4.0]:
            theta = diametro_para_nulo1(u1, 0.0)
            nulos = nulos_J0(theta, 0.0, m=1)
            assert abs(nulos[0] - u1) < 1e-6

    def test_ratio_Reff_monotono_decreciente(self):
        deltas = np.linspace(0, 1.1, 20)
        ratios = [ratio_Reff(d) for d in deltas]
        assert all(ratios[i] >= ratios[i+1] for i in range(len(ratios)-1))

    def test_reff_angular_proporcional(self):
        """Reff_angular debe ser proporcional a theta_g y a ratio_Reff."""
        from eht_tesis.modelo import Reff_angular
        for d in [0.0, 0.05, 0.1]:
            for th in [3.8, 4.0, 4.2]:
                Reff = Reff_angular(d, th)
                esperado = np.sqrt(27.0) * th * ratio_Reff(d)
                assert abs(Reff - esperado) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])