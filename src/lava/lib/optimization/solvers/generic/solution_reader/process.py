# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.process.ports.ports import InPort, RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.process.variable import Var


class SolutionReader(AbstractProcess):
    def __init__(
        self,
        target_costs,
        problem_index_map,
        num_in_ports: int = 1,
        var_shape=None,
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ):
        super().__init__(
            target_costs=target_costs,
            problem_index_map=problem_index_map,
            num_in_ports=num_in_ports,
            var_shape=var_shape,
            name=name,
            log_config=log_config,
        )
        # hard-coded very high cost
        min_cost = 2**24
        self.solution = Var(shape=var_shape, init=-1)
        self.solution_step = Var(shape=(1,), init=-1)
        self.min_cost = Var(shape=(2,), init=min_cost)
        self.cost = Var(shape=(1,), init=min_cost)
        self.satisfaction = Var(shape=(1,), init=0)
        self.ref_port = RefPort(shape=var_shape)
        for id in range(num_in_ports):
            setattr(self, f"read_gate_in_port_{id}", InPort(shape=(1,)))
