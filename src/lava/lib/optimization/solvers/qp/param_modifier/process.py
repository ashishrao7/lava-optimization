# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.process.ports.ports import RefPort
from lava.magma.core.process.process import AbstractProcess, LogConfig


class ParamModifier(AbstractProcess):
    """Process that triggers solution readout when problem is solved.

    Parameters
    ----------
    shape: The shape of the set of units in the downstream process whose state
        will be read by ParamModifers. The first entry of shape is the size of 
        alpha_man vector and the second entry is the size of beta_man vector.
    name: Name of the Process. Default is 'Process_ID', where ID is an
        integer value that is determined automatically.
    log_config: Configuration options for logging.

    RefPorts
    -------
    alpha_ref: RefPort to read/modify/write the value of alpha_man in cx_state 
    of PG neuron of the qp solver
    beta_ref: RefPort to read/modify/write the value of beta_man in cx_state of 
    PI neuron of the qp solver.
    """

    def __init__(self,
                 shape: ty.Tuple[int, ...],
                 name: ty.Optional[str] = None,
                 log_config: ty.Optional[LogConfig] = None) -> None:
        # how to handle shape here when two differnt shapes are required for 
        # the ports
        super().__init__(shape=shape,
                         name=name,
                         log_config=log_config)

        self.alpha_ref = RefPort(shape=(shape[0],))
        self.beta_ref = RefPort(shape=(shape[1],))
    