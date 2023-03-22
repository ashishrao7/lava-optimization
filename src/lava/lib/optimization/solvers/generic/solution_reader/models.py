# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.magma.core.decorator import implements
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


@implements(proc=SolutionReader, protocol=LoihiProtocol)
class SolutionReaderModel(AbstractSubProcessModel):
    def __init__(self, proc):

        target_costs = proc.proc_params.get("target_costs")
        problem_index_map = proc.proc_params.get("problem_index_map")
        num_in_ports = proc.proc_params.get("num_in_ports")
        var_shape = proc.proc_params.get("var_shape")

        self.read_gate = ReadGate(
            shape=var_shape, problem_index_map=problem_index_map, target_costs=target_costs, num_in_ports=num_in_ports,
        )
        self.solution_readout = SolutionReadout(
            shape=(,len(target_costs)), target_costs=target_costs, 
        )
        self.read_gate.cost_out.connect(self.solution_readout.cost_in)
        self.read_gate.solution_out.connect(self.solution_readout.read_solution)
        self.read_gate.send_pause_request.connect(
            self.solution_readout.timestep_in
        )

        proc.vars.solution.alias(self.solution_readout.solution)
        proc.vars.min_cost.alias(self.solution_readout.min_cost)
        proc.vars.solution_step.alias(self.solution_readout.solution_step)

        self.read_gate.solution_reader.connect(proc.ref_ports.ref_port)
        for id in range(num_in_ports):
            in_port = getattr(proc.in_ports, f"read_gate_in_port_{id}")
            out_port = getattr(self.read_gate, f"cost_in_{id}")
            in_port.connect(out_port)
