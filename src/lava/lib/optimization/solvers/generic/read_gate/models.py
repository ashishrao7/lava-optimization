# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np

from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

def init_proc(self, proc):
    super().__init__()
    self.target_costs_list = proc.proc_params.get("target_costs")
    self.problem_index_map = proc.proc_params.get("problem_index_map")

def readgate_post_guard(self):
    """Decide whether to run post management phase."""
    return True if self.min_cost else False


def readgate_run_spk(self):
    """Execute spiking phase, integrate input, update dynamics and
    send messages out."""
    in_ports = [
        port for port in self.py_ports if issubclass(type(port), PyInPort)
    ]
    min_prob_ids_and_cost = []
    curr_prob_ind=0
    port_num_start, port_num_end = 0, self.problem_index_map[curr_prob_ind]
    costs = []
    while port_num_start<port_num_end:
        if port_num_start==(self.problem_index_map[curr_prob_ind]):
            cost = min(costs)
            # ids of inports having min_cost according to the 
            # variable number of hyperparameters passed
            min_prob_ids_and_cost.append((port_num_start+costs.index(cost), cost))
            curr_prob_ind+=1
            costs = []
            if curr_prob_ind==len(self.problem_index_map):
                break
            else:
                port_num_start =  port_num_end
                port_num_end = self.problem_index_map[curr_prob_ind]
                continue
        costs.append(in_ports[port_num_start].recv()[0])
        port_num_start+=1

    # lengths of target costs and min_prob_ids should be the same
    for prob_num, target_cost in enumerate(self.target_costs_list):
        if self.solution is not None:
            timestep = -np.array([self.time_step])
            if self.min_cost <= target_cost:
                self._req_pause = True
            self.cost_out.send(np.array([min_prob_ids_and_cost[prob_num][1], self.min_prob_ids_and_cost[prob_num][0]]))
            self.send_pause_request.send(timestep)
            self.solution_out.send(self.solution)
            self.solution = None
            self.min_cost = None
            self.min_cost_id = None
        else:
            self.cost_out.send(np.array([0, 0]))
        if cost:
            self.min_cost_id = min_prob_ids_and_cost[prob_num][0]
            self.min_cost = min_prob_ids_and_cost[prob_num][1]


def readgate_run_post_mgmt(self):
    """Execute post management phase."""
    self.solution = self.solution_reader.read()


def get_readgate_members(num_in_ports):
    in_ports = {
        f"cost_in_{id}": LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                    precision=32)
        for id in range(num_in_ports)
    }
    readgate_members = {
        "target_cost": LavaPyType(int, np.int32, 32),
        "best_solution": LavaPyType(int, np.int32, 32),
        "cost_out": LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                               precision=32),
        "solution_out": LavaPyType(PyOutPort.VEC_DENSE, np.int32,
                                   precision=32),
        "send_pause_request": LavaPyType(
            PyOutPort.VEC_DENSE, np.int32, precision=32
        ),
        "solution_reader": LavaPyType(
            PyRefPort.VEC_DENSE, np.int32, precision=32
        ),
        "min_cost": None,
        "min_cost_id": None,
        "solution": None,
        "post_guard": readgate_post_guard,
        "__init__": init_proc,
        "run_spk": readgate_run_spk,
        "run_post_mgmt": readgate_run_post_mgmt,
    }
    readgate_members.update(in_ports)
    return readgate_members


def get_read_gate_model_class(num_in_ports: int):
    """Produce CPU model for the ReadGate process.

    The model verifies if better payload (cost) has been notified by the
    downstream processes, if so, it reads those processes state and sends
    out to
    the upstream process the new payload (cost) and the network state.
    """
    ReadGatePyModelBase = type(
        "ReadGatePyModel",
        (PyLoihiProcessModel,),
        get_readgate_members(num_in_ports),
    )
    ReadGatePyModelImpl = implements(ReadGate, protocol=LoihiProtocol)(
        ReadGatePyModelBase
    )
    ReadGatePyModel = requires(CPU)(ReadGatePyModelImpl)
    return ReadGatePyModel


ReadGatePyModel = get_read_gate_model_class(num_in_ports=1)

# probably new ReadGate model with multiple cost_in ports coming 
# in has to be written? Using multiple inports for cost will be slow no?
@implements(ReadGate, protocol=LoihiProtocol)
@requires(CPU)
class ReadGatePyModelDummy(PyLoihiProcessModel):
    """CPU model for the ReadGate process.

    The model verifies if better payload (cost) has been notified by the
    downstream processes, if so, it reads those processes state and sends out to
    the upstream process the new payload (cost) and the network state.
    """
    target_cost: int = LavaPyType(int, np.int32, 32)
    best_solution: int = LavaPyType(int, np.int32, 32)
    cost_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32,
                                   precision=32)
    cost_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_out: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    send_pause_request: PyOutPort = LavaPyType(
        PyOutPort.VEC_DENSE, np.int32, precision=32
    )
    solution_reader = LavaPyType(PyRefPort.VEC_DENSE, np.int32, precision=32)
    min_cost: int = None
    solution: np.ndarray = None

    def post_guard(self):
        """Decide whether to run post management phase."""
        return True if self.min_cost else False

    def run_spk(self):
        """Execute spiking phase, integrate input, update dynamics and
        send messages out."""
        cost = self.cost_in.recv()
        if cost[0]:
            self.min_cost = cost[0]
            self.cost_out.send(np.array([0]))
        elif self.solution is not None:
            timestep = - np.array([self.time_step])
            if self.min_cost <= self.target_cost:
                self._req_pause = True
            self.cost_out.send(np.array([self.min_cost]))
            self.send_pause_request.send(timestep)
            self.solution_out.send(self.solution)
            self.solution = None
            self.min_cost = None
        else:
            self.cost_out.send(np.array([0]))

    def run_post_mgmt(self):
        """Execute post management phase."""
        self.solution = self.solution_reader.read()
