# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

"""
Implement behaviors (models) of the processes defined in processes.py
For further documentation please refer to processes.py
"""
from ast import Del
import numpy as np
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.lib.optimization.solvers.qp.processes import (
    ConstraintDirections,
    ConstraintCheck,
    ConstraintNeurons,
    ConstraintNormals,
    QuadraticConnectivity,
    SolutionNeurons,
    GradientDynamics,
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
    SigmaNeurons,
    DeltaNeurons,
)


@implements(proc=ConstraintDirections, protocol=LoihiProtocol)
@requires(CPU)
class PyCDModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.col_sum = self.proc_params["col_sum"]

    def run_spk(self):
        s_in = self.s_in.recv()
        # Synops counter
        self.synops += np.sum(self.col_sum[s_in.nonzero()[0]])
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=ConstraintNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyCNeuModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    thresholds: np.ndarray = LavaPyType(np.ndarray, np.float64)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        a_in = self.a_in.recv()
        # process behavior: constraint violation check
        s_out = (a_in - self.thresholds) * (a_in > self.thresholds)
        # Spikeops counter
        self.spikeops += np.count_nonzero(s_out)
        self.s_out.send(s_out)


@implements(proc=QuadraticConnectivity, protocol=LoihiProtocol)
@requires(CPU)
class PyQCModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.col_sum = self.proc_params["col_sum"]

    def run_spk(self):
        s_in = self.s_in.recv()
        # Synops counter
        self.synops += np.sum(self.col_sum[s_in.nonzero()[0]])
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=SolutionNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySNModel(PyLoihiProcessModel):
    a_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    a_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_cc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_out = self.qp_neuron_state
        # Spikeops counter
        self.spikeops += np.count_nonzero(s_out)
        self.s_out_cc.send(s_out)
        self.s_out_qc.send(s_out)

        a_in_qc = self.a_in_qc.recv()
        a_in_cn = self.a_in_cn.recv()

        self.decay_counter += 1
        if self.decay_counter == self.alpha_decay_schedule:
            # TODO: guard against shift overflows in fixed-point
            self.alpha = np.right_shift(self.alpha, 1)
            self.decay_counter = np.zeros(self.decay_counter.shape)

        self.growth_counter += 1
        if self.growth_counter == self.beta_growth_schedule:
            self.beta = np.left_shift(self.beta, 1)
            # TODO: guard against shift overflows in fixed-point
            self.growth_counter = np.zeros(self.growth_counter.shape)

        # process behavior: gradient update
        self.qp_neuron_state += (
            -self.alpha * (a_in_qc + self.grad_bias) - self.beta * a_in_cn
        )


@implements(proc=ConstraintNormals, protocol=LoihiProtocol)
@requires(CPU)
class PyCNorModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    a_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    weights: np.ndarray = LavaPyType(np.ndarray, np.float64)

    # Profiling
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.col_sum = self.proc_params["col_sum"]

    def run_spk(self):
        s_in = self.s_in.recv()
        # Synops counter
        self.synops += np.sum(self.col_sum[s_in.nonzero()[0]])
        # process behavior: matrix multiplication
        a_out = self.weights @ s_in
        self.a_out.send(a_out)


@implements(proc=ConstraintCheck, protocol=LoihiProtocol)
class SubCCModel(AbstractSubProcessModel):
    """Implement constraintCheckProcess behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    constraint_matrix: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    # profiling
    cNeur_synops: int = LavaPyType(int, np.int32)
    cNeur_neurops: int = LavaPyType(int, np.int32)
    cNeur_spikeops: int = LavaPyType(int, np.int32)

    cD_synops: int = LavaPyType(int, np.int32)
    cD_neurops: int = LavaPyType(int, np.int32)
    cD_spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        constraint_matrix = proc.init_args.get("constraint_matrix", 0)
        constraint_bias = proc.init_args.get("constraint_bias", 0)

        x_int_init = proc.init_args.get("x_int_init", 0)
        sparse = proc.init_args.get("sparse", False)

        # Initialize subprocesses
        self.constraintDirections = ConstraintDirections(
            shape=constraint_matrix.shape,
            constraint_directions=constraint_matrix,
        )
        self.constraintNeurons = ConstraintNeurons(
            shape=constraint_bias.shape, thresholds=constraint_bias
        )

        if sparse:
            print("[INFO]: Using additional Sigma layer")
            self.sigmaNeurons = SigmaNeurons(
                shape=(constraint_matrix.shape[1], 1), x_sig_init=x_int_init
            )

            # proc.vars.x_internal.alias(self.sigmaNeurons.vars.x_internal)
            # connect subprocesses to obtain required process behavior
            proc.in_ports.s_in.connect(self.sigmaNeurons.in_ports.s_in)
            self.sigmaNeurons.out_ports.s_out.connect(
                self.constraintDirections.in_ports.s_in
            )

        else:
            proc.in_ports.s_in.connect(self.constraintDirections.in_ports.s_in)

        # remaining procesess to connect irrespective of sparsity
        self.constraintDirections.out_ports.a_out.connect(
            self.constraintNeurons.in_ports.a_in
        )
        self.constraintNeurons.out_ports.s_out.connect(proc.out_ports.s_out)

        # alias process variables to subprocess variables
        proc.vars.constraint_matrix.alias(
            self.constraintDirections.vars.weights
        )
        proc.vars.constraint_bias.alias(self.constraintNeurons.vars.thresholds)

        # profiling
        proc.vars.cNeur_synops.alias(self.constraintNeurons.vars.synops)
        proc.vars.cNeur_neurops.alias(self.constraintNeurons.vars.neurops)
        proc.vars.cNeur_spikeops.alias(self.constraintNeurons.vars.spikeops)
        proc.vars.cD_synops.alias(self.constraintDirections.vars.synops)
        proc.vars.cD_neurops.alias(self.constraintDirections.vars.neurops)
        proc.vars.cD_spikeops.alias(self.constraintDirections.vars.spikeops)


@implements(proc=GradientDynamics, protocol=LoihiProtocol)
class SubGDModel(AbstractSubProcessModel):
    """Implement gradientDynamics Process behavior via sub Processes."""

    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    hessian: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_matrix_T: np.ndarray = LavaPyType(
        np.ndarray,
        np.float64,
    )
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    # profiling
    cN_synops: int = LavaPyType(int, np.int32)
    cN_neurops: int = LavaPyType(int, np.int32)
    cN_spikeops: int = LavaPyType(int, np.int32)

    qC_synops: int = LavaPyType(int, np.int32)
    qC_neurops: int = LavaPyType(int, np.int32)
    qC_spikeops: int = LavaPyType(int, np.int32)

    sN_synops: int = LavaPyType(int, np.int32)
    sN_neurops: int = LavaPyType(int, np.int32)
    sN_spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc):
        """Builds sub Process structure of the Process."""
        hessian = proc.init_args.get("hessian", 0)
        shape_hess = hessian.shape
        shape_sol = (shape_hess[0], 1)
        constraint_matrix_T = proc.init_args.get("constraint_matrix_T", 0)
        shape_constraint_matrix_T = constraint_matrix_T.shape
        grad_bias = proc.init_args.get("grad_bias", np.zeros(shape_sol))
        qp_neuron_i = proc.init_args.get(
            "qp_neurons_init", np.zeros(shape_sol)
        )
        sparse = proc.init_args.get("sparse", False)
        model = proc.init_args.get("model", "SigDel")
        theta = proc.init_args.get("theta", np.zeros(shape_sol))
        vth_lo = proc.init_args.get("vth_lo", -10)
        vth_hi = proc.init_args.get("vth_hi", 10)
        alpha = proc.init_args.get("alpha", np.ones(shape_sol))
        beta = proc.init_args.get("beta", np.ones(shape_sol))
        t_d = proc.init_args.get("theta_decay_schedule", 100000)
        a_d = proc.init_args.get("alpha_decay_schedule", 100000)
        b_g = proc.init_args.get("beta_decay_schedule", 100000)

        # Initialize subprocesses
        self.qC = QuadraticConnectivity(shape=shape_hess, hessian=hessian)

        self.cN = ConstraintNormals(
            shape=shape_constraint_matrix_T,
            constraint_normals=constraint_matrix_T,
        )

        if sparse:
            if model == "SigDel":
                print("[INFO]: Using Sigma Delta Solution Neurons")
                self.sN = SolutionNeurons(
                    shape=shape_sol,
                    qp_neurons_init=qp_neuron_i,
                    grad_bias=grad_bias,
                    alpha=alpha,
                    beta=beta,
                    alpha_decay_schedule=a_d,
                    beta_growth_schedule=b_g,
                )

                self.sigmaNeurons = SigmaNeurons(
                    shape=shape_sol, x_sig_init=qp_neuron_i
                )
                # proc.vars.x_internal_sigma.alias(
                #     self.sigmaNeurons.vars.x_internal
                # )

                self.deltaNeurons = DeltaNeurons(
                    shape=shape_sol,
                    x_del_init=qp_neuron_i,
                    theta=theta,
                    theta_decay_schedule=t_d,
                )
                # proc.vars.x_internal_delta.alias(
                #     self.deltaNeurons.vars.x_internal
                # )
                proc.vars.theta.alias(self.deltaNeurons.vars.theta)
                proc.vars.theta_decay_schedule.alias(
                    self.deltaNeurons.vars.theta_decay_schedule
                )

                # connection processes and aliases
                self.sN.out_ports.s_out_qc.connect(
                    self.deltaNeurons.in_ports.s_in
                )
                self.deltaNeurons.out_ports.s_out.connect(
                    self.sigmaNeurons.in_ports.s_in
                )
                self.deltaNeurons.out_ports.s_out.connect(proc.out_ports.s_out)
                self.sigmaNeurons.out_ports.s_out.connect(
                    self.qC.in_ports.s_in
                )
                proc.vars.sN_spikeops.alias(self.deltaNeurons.vars.spikeops)

        else:
            print("[INFO]: Using Dense Solution Neurons")
            self.sN = SolutionNeurons(
                shape=shape_sol,
                qp_neurons_init=qp_neuron_i,
                grad_bias=grad_bias,
                alpha=alpha,
                beta=beta,
                alpha_decay_schedule=a_d,
                beta_growth_schedule=b_g,
            )
            self.sN.out_ports.s_out_qc.connect(self.qC.in_ports.s_in)
            self.sN.out_ports.s_out_cc.connect(proc.out_ports.s_out)
            proc.vars.sN_spikeops.alias(self.sN.vars.spikeops)

        # connect subprocesses to obtain required process behavior
        proc.in_ports.s_in.connect(self.cN.in_ports.s_in)
        self.cN.out_ports.a_out.connect(self.sN.in_ports.a_in_cn)
        self.qC.out_ports.a_out.connect(self.sN.in_ports.a_in_qc)

        # alias process variables to subprocess variables
        proc.vars.hessian.alias(self.qC.vars.weights)
        proc.vars.constraint_matrix_T.alias(self.cN.vars.weights)
        proc.vars.grad_bias.alias(self.sN.vars.grad_bias)
        proc.vars.qp_neuron_state.alias(self.sN.vars.qp_neuron_state)
        proc.vars.alpha.alias(self.sN.vars.alpha)
        proc.vars.beta.alias(self.sN.vars.beta)
        proc.vars.alpha_decay_schedule.alias(self.sN.vars.alpha_decay_schedule)
        proc.vars.beta_growth_schedule.alias(self.sN.vars.beta_growth_schedule)

        # profiling
        proc.vars.cN_synops.alias(self.cN.vars.synops)
        proc.vars.cN_neurops.alias(self.cN.vars.neurops)
        proc.vars.cN_spikeops.alias(self.cN.vars.spikeops)

        proc.vars.qC_synops.alias(self.qC.vars.synops)
        proc.vars.qC_neurops.alias(self.qC.vars.neurops)
        proc.vars.qC_spikeops.alias(self.qC.vars.spikeops)

        proc.vars.sN_synops.alias(self.sN.vars.synops)
        proc.vars.sN_neurops.alias(self.sN.vars.neurops)


@implements(proc=ProjectedGradientNeuronsPIPGeq, protocol=LoihiProtocol)
@requires(CPU)
class PyProjGradPIPGeqModel(PyLoihiProcessModel):
    a_in_qc: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_qc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    a_in_cn: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out_cd: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    qp_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    grad_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha: np.ndarray = LavaPyType(np.ndarray, np.float64)
    alpha_decay_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.lr_decay_type = self.proc_params["lr_decay_type"]
        self.alpha_decay_indices = self.proc_params["alpha_decay_indices"]

    def run_spk(self):
        s_out = self.qp_neuron_state
        # Spikops counter
        self.spikeops += np.count_nonzero(s_out)
        self.s_out_cd.send(s_out)
        self.s_out_qc.send(s_out)

        a_in_qc = self.a_in_qc.recv()
        a_in_cn = self.a_in_cn.recv()
        self.decay_counter += 1
        if self.lr_decay_type == "schedule":
            if self.decay_counter == self.alpha_decay_schedule:
                # TODO: guard against shift overflows in fixed-point
                self.alpha = self.alpha / 2
                self.decay_counter = np.zeros(self.decay_counter.shape)
        if self.lr_decay_type == "indices":
            if self.decay_counter in self.alpha_decay_indices:
                self.alpha = self.alpha / 2

        # process behavior: gradient update
        self.qp_neuron_state -= self.alpha * (
            a_in_qc + self.grad_bias + a_in_cn
        )


@implements(proc=ProportionalIntegralNeuronsPIPGeq, protocol=LoihiProtocol)
@requires(CPU)
class PyPIneurPIPGeqModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    constraint_neuron_state: np.ndarray = LavaPyType(np.ndarray, np.float64)
    constraint_bias: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    beta_growth_schedule: int = LavaPyType(int, np.int32)
    growth_counter: int = LavaPyType(int, np.int32)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.lr_growth_type = self.proc_params["lr_growth_type"]
        self.beta_growth_indices = self.proc_params["beta_growth_indices"]

    def run_spk(self):
        a_in = self.a_in.recv()
        self.growth_counter += 1
        if self.lr_growth_type == "schedule":
            if self.growth_counter == self.beta_growth_schedule:
                # TODO: guard against shift overflows in fixed-point
                self.beta = self.beta * 2
                self.growth_counter = np.zeros(self.growth_counter.shape)
        if self.lr_growth_type == "indices":
            if self.growth_counter in self.beta_growth_indices:
                self.beta = self.beta * 2

        # process behavior:
        omega = self.beta * (a_in - self.constraint_bias)
        self.constraint_neuron_state += omega
        gamma = self.constraint_neuron_state + omega
        # Spikeops counter
        self.spikeops += np.count_nonzero(gamma)
        self.s_out.send(gamma)


@implements(proc=SigmaNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PySigNeurModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def run_spk(self):
        s_in = self.s_in.recv()
        self.x_internal += s_in
        s_out = self.x_internal
        self.s_out.send(s_out)


@implements(proc=DeltaNeurons, protocol=LoihiProtocol)
@requires(CPU)
class PyDelNeurModel(PyLoihiProcessModel):
    s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    x_internal: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta: np.ndarray = LavaPyType(np.ndarray, np.float64)
    theta_decay_schedule: int = LavaPyType(int, np.int32)
    decay_counter: int = LavaPyType(int, np.int32)

    # Profiling Vars
    synops: int = LavaPyType(int, np.int32)
    neurops: int = LavaPyType(int, np.int32)
    spikeops: int = LavaPyType(int, np.int32)

    def __init__(self, proc_params: dict) -> None:
        super().__init__(proc_params)
        self.theta_decay_type = self.proc_params["theta_decay_type"]
        self.theta_decay_indices = self.proc_params["theta_decay_indices"]

    def run_spk(self):
        s_in = self.s_in.recv()
        delta_state = s_in - self.x_internal
        self.x_internal = s_in
        self.decay_counter += 1
        if self.theta_decay_type == "schedule":
            if self.decay_counter == self.theta_decay_schedule:
                # TODO: guard against shift overflows in fixed-point
                self.theta = self.theta / 2
                self.decay_counter = np.zeros(self.decay_counter.shape)
        if self.theta_decay_type == "indices":
            if self.decay_counter in self.theta_decay_indices:
                self.theta = self.theta / 2
        s_out = delta_state * (np.abs(delta_state) >= self.theta)
        # Spikeops counter
        self.spikeops += np.count_nonzero(s_out)
        self.s_out.send(s_out)
