# -*- coding: utf-8 -*-
# File: utils/lr_lambda.py

from collections.abc import Callable
from typing import Any

import numpy as np


class LRLambda(object):
    def __init__(
        self,
        lr_lambda: Callable[[int, int, float, float], float],
        epochs: int,
        max_lr: float,
        min_lr: float = 1e-6,
        warmup: int = 0,
        early_min: int = 0,
    ) -> None:
        assert epochs > 0
        assert max_lr > 0
        assert min_lr > 0
        assert max_lr >= min_lr
        assert warmup >= 0
        assert early_min >= 0
        assert warmup + early_min < epochs

        self.__lr_lambda = lr_lambda
        self.__max_lr = max_lr
        self.__min_lr = min_lr
        self.__warmup = warmup
        self.__last_epoch = epochs - early_min - 1
        self.__lambda_epochs = epochs - 1 - warmup - early_min
        self.__warmup_slope = 0 if warmup == 0 else (max_lr - min_lr) / warmup

    def __call__(self, epoch: int) -> float:
        assert epoch >= 0

        if self.__warmup <= epoch < self.__last_epoch:
            lr = self.__lr_lambda(
                epoch - self.__warmup,
                self.__lambda_epochs,
                self.__max_lr,
                self.__min_lr,
            )
        elif epoch < self.__warmup:  # Linear
            lr = self.__warmup_slope * epoch + self.__min_lr
        else:  # Constant
            lr = self.__min_lr

        return lr

    @classmethod
    def linear(cls, *args: Any, **kwargs: Any) -> "LRLambda":
        return cls(
            lambda epoch, epochs, max_lr, min_lr: (min_lr - max_lr) / epochs * epoch + max_lr,
            *args,
            **kwargs,
        )

    @classmethod
    def cosine(cls, *args: Any, **kwargs: Any) -> "LRLambda":
        return cls(
            lambda epoch, epochs, max_lr, min_lr: ((max_lr - min_lr) / 2) * (np.cos((epoch / epochs) * np.pi) + 1)
            + min_lr,
            *args,
            **kwargs,
        )
