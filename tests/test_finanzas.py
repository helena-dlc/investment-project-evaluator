import math
import unittest

import finanzas as fz


class TestFinanzas(unittest.TestCase):
    def setUp(self):
        self.flujos = [-100_000, 30_000, 35_000, 40_000, 45_000]
        self.tasa = 0.10

    def test_van_del_ejemplo(self):
        self.assertAlmostEqual(fz.van(self.flujos, self.tasa), 16_986.544634929287, places=8)

    def test_van_detallado_reconcilia_con_total(self):
        detalle = fz.van_detallado(self.flujos, self.tasa)
        self.assertAlmostEqual(detalle[-1]["van_acumulado"], fz.van(self.flujos, self.tasa))
        self.assertEqual(detalle[0]["periodo"], 0)
        self.assertEqual(detalle[0]["factor_descuento"], 1)

    def test_tir_convencional_es_unica(self):
        resultado = fz.tir(self.flujos)
        self.assertIsNotNone(resultado)
        self.assertAlmostEqual(resultado, 0.17093686339499115, places=10)
        self.assertAlmostEqual(fz.van(self.flujos, resultado), 0, places=5)
        self.assertTrue(fz.tir_es_confiable(self.flujos))

    def test_detecta_dos_tir_y_no_elige_una_arbitrariamente(self):
        flujos = [-1_000, 5_000, -6_000]
        self.assertEqual(len(fz.tirs(flujos)), 2)
        self.assertAlmostEqual(fz.tirs(flujos)[0], 1.0)
        self.assertAlmostEqual(fz.tirs(flujos)[1], 2.0)
        self.assertIsNone(fz.tir(flujos))
        self.assertFalse(fz.tir_es_confiable(flujos))

    def test_payback_interpola_dentro_del_periodo(self):
        self.assertAlmostEqual(fz.payback(self.flujos), 2.875)
        self.assertAlmostEqual(fz.payback(self.flujos, self.tasa), 3.447333333333334, places=10)

    def test_payback_devuelve_none_si_no_recupera(self):
        self.assertIsNone(fz.payback([-100_000, 5_000, 5_000, 5_000]))

    def test_indice_incluye_egresos_futuros(self):
        flujos = [-100_000, 60_000, -20_000, 90_000]
        esperado = sum(
            flujo / (1 + self.tasa) ** periodo
            for periodo, flujo in enumerate(flujos)
            if periodo > 0
        ) / 100_000
        self.assertAlmostEqual(fz.indice_rentabilidad(flujos, self.tasa), esperado)
        self.assertAlmostEqual(esperado, 1.0563486, places=6)

    def test_vae_con_tasa_cero_usa_limite_del_frc(self):
        self.assertAlmostEqual(fz.factor_recuperacion_capital(0, 4), 0.25)
        self.assertAlmostEqual(fz.vae(self.flujos, 0), 12_500)

    def test_evaluacion_consolida_resultados_y_advertencias(self):
        evaluacion = fz.evaluar([-1_000, 5_000, -6_000], self.tasa)
        self.assertIsNone(evaluacion.tir)
        self.assertEqual(len(evaluacion.tirs), 2)
        self.assertAlmostEqual(evaluacion.tirs[0], 1.0)
        self.assertAlmostEqual(evaluacion.tirs[1], 2.0)
        self.assertFalse(evaluacion.tir_confiable)
        self.assertTrue(any("varias TIR" in aviso for aviso in evaluacion.advertencias))

    def test_valida_tasas_y_valores_no_finitos(self):
        for tasa in (-1, -1.1, math.inf, math.nan):
            with self.subTest(tasa=tasa):
                with self.assertRaises(ValueError):
                    fz.van(self.flujos, tasa)
        with self.assertRaises(ValueError):
            fz.van([-100, math.inf], self.tasa)


if __name__ == "__main__":
    unittest.main()
