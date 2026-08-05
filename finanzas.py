"""Motor financiero independiente de la interfaz de usuario."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Sequence

import numpy as np


def _flujos_validos(flujos: Sequence[float]) -> tuple[float, ...]:
    """Normaliza los flujos y rechaza valores que invalidan los cálculos."""
    valores = tuple(float(flujo) for flujo in flujos)
    if not valores:
        raise ValueError("Se requiere al menos un flujo de caja.")
    if not all(isfinite(flujo) for flujo in valores):
        raise ValueError("Los flujos de caja deben ser números finitos.")
    return valores


def _tasa_valida(tasa: float) -> float:
    tasa = float(tasa)
    if not isfinite(tasa) or tasa <= -1:
        raise ValueError("La tasa debe ser un número finito mayor a -100%.")
    return tasa


def van(flujos: Sequence[float], tasa: float) -> float:
    """Calcula el VAN; ``flujos[0]`` corresponde al período cero."""
    valores = _flujos_validos(flujos)
    tasa = _tasa_valida(tasa)
    return sum(flujo / (1 + tasa) ** periodo for periodo, flujo in enumerate(valores))


def van_detallado(flujos: Sequence[float], tasa: float) -> list[dict[str, float | int]]:
    """Desarrolla el VAN período a período."""
    valores = _flujos_validos(flujos)
    tasa = _tasa_valida(tasa)
    filas: list[dict[str, float | int]] = []
    acumulado = 0.0

    for periodo, flujo in enumerate(valores):
        factor = (1 + tasa) ** periodo
        valor_presente = flujo / factor
        acumulado += valor_presente
        filas.append(
            {
                "periodo": periodo,
                "flujo": flujo,
                "factor_descuento": factor,
                "valor_presente": valor_presente,
                "van_acumulado": acumulado,
            }
        )
    return filas


def _cambios_de_signo(flujos: Sequence[float]) -> int:
    signos = [1 if flujo > 0 else -1 for flujo in _flujos_validos(flujos) if flujo != 0]
    return sum(anterior != siguiente for anterior, siguiente in zip(signos, signos[1:]))


def _refinar_tir(flujos: tuple[float, ...], tasa: float) -> float:
    """Mejora una raíz obtenida del polinomio mediante Newton acotado."""
    for _ in range(30):
        base = 1 + tasa
        if base <= 0:
            break
        valor = sum(flujo / base**periodo for periodo, flujo in enumerate(flujos))
        derivada = sum(
            -periodo * flujo / base ** (periodo + 1)
            for periodo, flujo in enumerate(flujos)
            if periodo
        )
        if not isfinite(valor) or not isfinite(derivada) or abs(derivada) < 1e-14:
            break
        siguiente = tasa - valor / derivada
        if siguiente <= -1 or not isfinite(siguiente):
            break
        if abs(siguiente - tasa) <= 1e-12 * max(1.0, abs(tasa)):
            return siguiente
        tasa = siguiente
    return tasa


def tirs(
    flujos: Sequence[float],
    tasa_maxima: float | None = 10.0,
    tolerancia_imaginaria: float = 1e-7,
) -> list[float]:
    """
    Devuelve todas las TIR reales detectadas en el dominio solicitado.

    La ecuación del VAN se transforma en un polinomio en ``x = 1 + tasa``.
    Así se detectan varias TIR en vez de depender de un punto inicial.
    """
    valores = _flujos_validos(flujos)
    if len(valores) < 2 or not any(valores):
        return []
    if tasa_maxima is not None and tasa_maxima <= -1:
        raise ValueError("La tasa máxima debe ser mayor a -100%.")

    # VAN = f0 + f1/x + ... + fn/x^n. Al multiplicar por x^n,
    # los flujos son directamente los coeficientes del polinomio.
    raices_x = np.roots(np.asarray(valores, dtype=float))
    candidatas: list[float] = []
    for raiz in raices_x:
        escala = max(1.0, abs(float(raiz.real)))
        if abs(float(raiz.imag)) > tolerancia_imaginaria * escala or raiz.real <= 0:
            continue
        tasa = _refinar_tir(valores, float(raiz.real) - 1)
        if tasa <= -1 or (tasa_maxima is not None and tasa > tasa_maxima):
            continue
        if abs(van(valores, tasa)) > 1e-5 * max(1.0, max(abs(f) for f in valores)):
            continue
        if not any(abs(tasa - previa) <= 1e-7 * max(1.0, abs(tasa)) for previa in candidatas):
            candidatas.append(tasa)

    return sorted(candidatas)


def tir(flujos: Sequence[float], tasa_maxima: float | None = 10.0) -> float | None:
    """Devuelve la TIR solamente cuando existe una única raíz real admisible."""
    raices = tirs(flujos, tasa_maxima=tasa_maxima)
    return raices[0] if len(raices) == 1 else None


def tir_es_confiable(flujos: Sequence[float]) -> bool:
    """Indica si un único cambio de signo garantiza una TIR no ambigua."""
    return _cambios_de_signo(flujos) == 1


def payback(flujos: Sequence[float], tasa: float | None = None) -> float | None:
    """Calcula el payback interpolado, simple o descontado."""
    valores = _flujos_validos(flujos)
    if tasa is not None:
        tasa = _tasa_valida(tasa)

    acumulado = 0.0
    for periodo, flujo in enumerate(valores):
        valor = flujo / (1 + tasa) ** periodo if tasa is not None else flujo
        anterior = acumulado
        acumulado += valor
        if acumulado >= 0:
            if periodo == 0:
                return 0.0
            if valor == 0:
                return float(periodo)
            return (periodo - 1) + (-anterior / valor)
    return None


def indice_rentabilidad(flujos: Sequence[float], tasa: float) -> float:
    """
    Calcula VP de todos los flujos futuros / inversión inicial absoluta.

    Los egresos futuros también se descuentan; omitirlos infla el indicador.
    """
    valores = _flujos_validos(flujos)
    tasa = _tasa_valida(tasa)
    inversion = abs(valores[0])
    if inversion == 0:
        return float("inf")
    vp_futuros = sum(
        flujo / (1 + tasa) ** periodo
        for periodo, flujo in enumerate(valores)
        if periodo > 0
    )
    return vp_futuros / inversion


def factor_recuperacion_capital(tasa: float, periodos: int) -> float:
    """FRC = r(1+r)^n / ((1+r)^n - 1), con su límite 1/n para r=0."""
    tasa = _tasa_valida(tasa)
    if periodos <= 0:
        raise ValueError("El número de períodos debe ser mayor a cero.")
    if tasa == 0:
        return 1 / periodos
    potencia = (1 + tasa) ** periodos
    return tasa * potencia / (potencia - 1)


def vae(flujos: Sequence[float], tasa: float) -> float:
    """Convierte el VAN en una anualidad uniforme equivalente."""
    valores = _flujos_validos(flujos)
    periodos = len(valores) - 1
    if periodos <= 0:
        return 0.0
    return van(valores, tasa) * factor_recuperacion_capital(tasa, periodos)


caue = vae


@dataclass
class Evaluacion:
    van: float
    tir: float | None
    tir_confiable: bool
    payback: float | None
    payback_descontado: float | None
    indice_rentabilidad: float
    vae: float
    tirs: list[float] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    @property
    def viable(self) -> bool:
        return self.van > 0


def evaluar(flujos: Sequence[float], tasa: float) -> Evaluacion:
    """Calcula los indicadores y devuelve advertencias metodológicas."""
    valores = _flujos_validos(flujos)
    tasa = _tasa_valida(tasa)
    if len(valores) < 2:
        raise ValueError("Se requieren al menos dos períodos.")

    avisos: list[str] = []
    confiable = tir_es_confiable(valores)
    raices = tirs(valores)
    tir_unica = raices[0] if len(raices) == 1 else None

    if not confiable:
        avisos.append(
            "Los flujos no son convencionales: la TIR puede ser ambigua. "
            "Use el VAN como criterio principal."
        )
    if not raices:
        avisos.append("No existe TIR real en el rango -100% a 1000%.")
    elif len(raices) > 1:
        porcentajes = ", ".join(f"{raiz:.2%}" for raiz in raices)
        avisos.append(
            f"Se detectaron varias TIR ({porcentajes}); no se informa una TIR única."
        )
    elif tir_unica is not None and tir_unica > 1:
        avisos.append(
            f"La TIR ({tir_unica:.1%}) es inusualmente alta; revise los supuestos."
        )
    if valores[0] >= 0:
        avisos.append(
            "El flujo del período 0 no es negativo; el índice de rentabilidad "
            "y el payback pierden sentido económico."
        )

    return Evaluacion(
        van=van(valores, tasa),
        tir=tir_unica,
        tir_confiable=confiable,
        payback=payback(valores),
        payback_descontado=payback(valores, tasa),
        indice_rentabilidad=indice_rentabilidad(valores, tasa),
        vae=vae(valores, tasa),
        tirs=raices,
        advertencias=avisos,
    )
