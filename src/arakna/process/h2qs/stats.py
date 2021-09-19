#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    r2_score,
)

T = t.TypeVar("T")


#####
# Вспомогательный класс для описания отдельной метрики качества
#####


class Property(t.Generic[T]):
    def __init__(
        self,
        df: pd.DataFrame,
        value_attr: str,
        convert: t.Callable[[t.Any], T],
        *,
        hurst_attr: str = "hurst",
    ) -> None:
        self.__df = df
        self.__value_attr = value_attr
        self.__convert = convert
        self.__hurst_attr = hurst_attr

    def min(self) -> T:
        return self.__convert(self.__df[self.__value_attr].min())

    def max(self) -> T:
        return self.__convert(self.__df[self.__value_attr].max())

    def avg(self) -> float:
        h_curr = self.__df[self.__hurst_attr].values

        h_next = np.ones_like(h_curr)
        h_next[:-1] = h_curr[1:]

        return float(2.0 * (self.__df[self.__value_attr].values @ (h_next - h_curr)))

    def to_dict(self) -> t.Dict[str, t.Union[T, float]]:
        return {
            "min": self.min(),
            "max": self.max(),
            "avg": self.avg(),
        }

    @property
    def name(self):
        return self.__value_attr

    @property
    def values(self) -> np.ndarray:
        return self.__df[self.__value_attr].values.flatten()


#####
# Вспомогательный класс для определения метрик качества
# при оценке моделей предсказния размера очереди
#####


class Stats:
    def __init__(
        self,
        h_values: t.Iterable[float],
        qs_true: t.Iterable[int],
        qs_pred: t.Iterable[float],
        population_size: t.Iterable[int],
        beta_plus: float,
        beta_minus: float,
        delta: int,
    ) -> None:
        h_array = np.fromiter(h_values, dtype=float)
        index = range(1, h_array.size + 1)
        h_series = pd.Series(h_array, index=index)

        ps_series = pd.Series(np.fromiter(population_size, dtype=int), index=index)
        qs_true_series = pd.Series(np.fromiter(qs_true, dtype=int), index=index)

        qs_pred_array = np.fromiter(qs_pred, dtype=float)
        qs_pred_series = pd.Series(qs_pred_array, index=index)

        self.__df = (
            pd.concat(
                {
                    "hurst": h_series,
                    "population_size": ps_series,
                    "qs_true": qs_true_series,
                    "qs_pred": qs_pred_series,
                },
                axis=1,
            )
            .dropna()
            .sort_values(by=["hurst"])
        )

        self.__df = self.__df[
            (self.__df["hurst"] >= 0.5)
            & (self.__df["hurst"] <= 1.0)
            & (self.__df["qs_true"] >= 0)
            & (self.__df["population_size"] > 0)
        ]

        self.__df["qs_pred"] = np.ceil(self.__df["qs_pred"]).astype(int)

        # Получить значение количества потерянных пакетов.
        #
        # Пусть $y_i$ - истинное значение размера очереди в выборке,
        # $\hat{y}_i$ - предсказанное значение размера очереди в выборке. Если
        # $y_i > \hat{y}_i$, то в системе могут возникнуть потери в размере
        # $l_i = y_i - \hat{y}_i$ пакетов. В противном случае, при преобразовании
        # трафика не возникнет потерь пакетов, поскольку предсказанный размер очереди
        # больше того, что требуется при преобразовании трафика.
        #
        # При обучении модели с использованием бесконечной очереди такие потери носят
        # характер предполагаемых потерь.
        self.__df["loss"] = np.where(
            self.__df["qs_true"] > self.__df["qs_pred"],
            self.__df["qs_true"] - self.__df["qs_pred"],
            0,
        )

        self.__df["loss_probability"] = self.__df["loss"] / self.__df["population_size"]
        self.__df["qos"] = 1.0 - self.__df["loss_probability"]

        # Получить значение эффективности использования очереди.
        #
        # Пусть $y_i$ - истинное значение размера очереди в выборке,
        # $\hat{y}_i$ - предсказанное значение размера очереди в выборке.
        # Если $y_i < \hat{y}_i$, то эффективность использования очереди может
        # быть получена из соотношения:
        #
        # $$e_i = \frac{y_i}{\hat{y}_i}.$$
        #
        # В противном случае очередь будет заполнена полностью и эффективность
        # использования очереди $e_i = 1$.
        self.__df["queue_efficiency"] = np.where(
            (self.__df["qs_true"] < self.__df["qs_pred"]) & (self.__df["qs_pred"] != 0),
            self.__df["qs_true"] / self.__df["qs_pred"],
            1.0,
        )

        # Получить среднее значение величины штрафа.
        #
        # Пусть $y_i$ - истинное значение размера очереди в выборке,
        # $\hat{y}_i$ - предсказанное значение размера очереди в выборке,
        # соответствующее истинному значению $y_i$. Если $y_i > \hat{y}_i$,
        # то будем штрафовать обучаемую систему на величину
        # $\beta_{+}\cdot(y_i - \hat{y}_i)$. Если же $\hat{y}_i \ge y_i$,
        # то величина штрафа будет зависеть от величины разности
        # $\Delta_i = \hat{y}_i - y_i$, при $\Delta_i > \Delta$ величина штрафа
        # составит $\beta_{-}\cdot(\Delta_i - \Delta)$ и 0 - в противном случае.
        # Таким образом, для заданных $y_i$ и $\hat{y}_i$ величина штрафа будет
        # определяться из равенства:
        #
        # $$
        # p_i = \begin{cases}
        #          \beta_{+}\cdot(y_i - \hat{y}_i), & \text{если } y_i > \hat{y}_i,\\
        #          \beta_{-}\cdot(\hat{y}_i - y_i - \Delta), & \text{если }
        #             \hat{y}_i - y_i > \Delta,\\
        #          0, & \text{в противном случае}.\\
        #      \end{cases}
        # $$
        #
        # Общий штраф для всех испытаний будет определяться как среднее
        # арифметическое между штрафами по каждому испытанию, полученными
        # по предыдущей формуле:
        #
        # $$p = \frac 1n \sum_{i = 1}^n p_i,$$
        #
        # где $n$ - количество испытаний.
        #
        # Представленная система штрафов предусматривает введение трех
        # гиперпараметров: $\beta_{+}$, $\beta_{-}$ и $\Delta$.
        # Где $\beta_{+}\ge\beta_{-} > 0$ и $\Delta > 0$.
        self.__df["penalty"] = self.penalty_scores(
            self.__df["qs_true"].values,
            self.__df["qs_pred"].values,
            beta_plus,
            beta_minus,
            delta,
        )

        self.__loss: t.Optional[Property[int]] = None
        self.__loss_probability: t.Optional[Property[float]] = None
        self.__qos: t.Optional[Property[float]] = None
        self.__queue_efficiency: t.Optional[Property[float]] = None
        self.__penalty: t.Optional[Property[float]] = None

    def __iter__(self) -> t.Generator[Property[t.Any], None, None]:
        yield self.loss
        yield self.loss_probability
        yield self.qos
        yield self.queue_efficiency
        yield self.penalty

    @property
    def loss(self) -> Property[int]:
        if self.__loss is None:
            self.__loss = Property(self.__df, "loss", int)

        return self.__loss

    @property
    def loss_probability(self) -> Property[float]:
        if self.__loss_probability is None:
            self.__loss_probability = Property(self.__df, "loss_probability", float)

        return self.__loss_probability

    @property
    def qos(self) -> Property[float]:
        if self.__qos is None:
            self.__qos = Property(self.__df, "qos", float)

        return self.__qos

    @property
    def queue_efficiency(self) -> Property[float]:
        if self.__queue_efficiency is None:
            self.__queue_efficiency = Property(self.__df, "queue_efficiency", float)

        return self.__queue_efficiency

    @property
    def penalty(self) -> Property[float]:
        if self.__penalty is None:
            self.__penalty = Property(self.__df, "penalty", float)

        return self.__penalty

    def to_parquet(self, path: Path) -> None:
        self.__df.to_parquet(str(path), engine="auto")

    def avg(self) -> pd.Series:
        return pd.Series(
            {
                self.loss.name: self.loss.avg(),
                self.loss_probability.name: self.loss_probability.avg(),
                self.qos.name: self.qos.avg(),
                self.queue_efficiency.name: self.queue_efficiency.avg(),
                self.penalty.name: self.penalty.avg(),
            }
        )

    @classmethod
    def penalty_scores(
        cls,
        qs_true: np.ndarray,
        qs_pred: np.ndarray,
        beta_plus: float,
        beta_minus: float,
        delta: int,
    ) -> np.ndarray:
        qs_true = qs_true.flatten()
        qs_pred = qs_pred.flatten()

        return np.where(
            qs_true > qs_pred,
            beta_plus * (qs_true - qs_pred),
            np.where(
                qs_pred - qs_true > delta, beta_minus * (qs_pred - qs_true - delta), 0.0
            ),
        )

    @classmethod
    def make_penalty_scorer(
        cls, beta_plus: float, beta_minus: float, delta: int
    ) -> t.Any:
        def penalty_score(qs_true: np.ndarray, qs_pred: np.ndarray) -> float:
            return float(
                np.mean(
                    cls.penalty_scores(qs_true, qs_pred, beta_plus, beta_minus, delta)
                )
            )

        return make_scorer(penalty_score, greater_is_better=False)

    @staticmethod
    def metrics(qs_true: np.ndarray, qs_pred: np.ndarray) -> t.Dict[str, float]:
        return {
            "R2": r2_score(qs_true, qs_pred),
            "RE": max_error(qs_true, qs_pred),
            "RMSE": np.sqrt(mean_squared_error(qs_true, qs_pred)),
            "MAE": mean_absolute_error(qs_true, qs_pred),
            "MAPE": mean_absolute_percentage_error(qs_true, qs_pred),
            "MSLE": mean_squared_log_error(qs_true, qs_pred),
            "MED": median_absolute_error(qs_true, qs_pred),
            "EVS": explained_variance_score(qs_true, qs_pred),
            "MTD": mean_tweedie_deviance(qs_true, qs_pred),
            "MPD": mean_poisson_deviance(qs_true, qs_pred),
            "MGD": mean_gamma_deviance(qs_true, qs_pred),
        }
