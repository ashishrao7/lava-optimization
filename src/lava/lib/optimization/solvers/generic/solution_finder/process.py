# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

import numpy as np
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class SolutionFinder(AbstractProcess):
    def __init__(
        self,
        problem,
        hyperparameters,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ):
        super().__init__(
            problem,
            hyperparameters=hyperparameters,
            name=name,
            log_config=log_config,
        )
        self.variables_assignment = Var(shape=problem.discrete_variables.shape, init=(1,))
        self.cost = Var(shape=(1,), init=(1,))
        self.cost_out = OutPort(shape=(1,))
