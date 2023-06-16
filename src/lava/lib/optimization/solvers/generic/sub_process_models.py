# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

try:
    from lava.magma.core.model.nc.net import NetL2
except ImportError:

    class NetL2:
        pass

import numpy as np
from lava.lib.optimization.solvers.generic.cost_integrator.process import \
    CostIntegrator
from lava.lib.optimization.solvers.generic.hierarchical_processes import \
    CostConvergenceChecker, DiscreteVariablesProcess, \
    ContinuousVariablesProcess, ContinuousConstraintsProcess, \
    StochasticIntegrateAndFire, NEBMAbstract, \
    NEBMSimulatedAnnealingAbstract

from lava.lib.optimization.solvers.generic.nebm.process import NEBM, \
    NEBMSimulatedAnnealing
from lava.lib.optimization.solvers.generic.scif.process import QuboScif
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import Loihi2NeuroCore, CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.lib.optimization.solvers.qp.models import (
    ProjectedGradientNeuronsPIPGeq,
    ProportionalIntegralNeuronsPIPGeq,
)
from lava.proc.sparse.process import Sparse
from lava.lib.optimization.utils.qp_processing import convert_to_fp

@implements(proc=ContinuousVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class ContinuousVariablesModel(AbstractSubProcessModel):
    """Model for the ContinuousVariables process.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        neuron_model = proc.hyperparameters.get("neuron_model", "qp/lp-pipg")
        
        if neuron_model == "qp/lp-pipg":
            # these are part of the problem not hyperparams
            # adding them here to show that they are need for the neurons models
            # since some values are calculated based on these weights 
            p_pre = proc.hyperparameters.get("p", 1)
            A_pre = proc.hyperparameters.get("A", 0)
            Q_pre = proc.hyperparameters.get("Q", 0)
            # legitimate hyperparameters
            init_state = proc.hyperparameters.get("init_state",
                                                  np.zeros((p_pre.shape[0],), 
                                                           dtype=int))
            
            alpha_man =  proc.hyperparameters.get("alpha_mantissa", 1)
            alpha_exp =  proc.hyperparameters.get("alpha_exponent", 1)
            decay_params = proc.hyperparameters.get("decay_schedule_parameters", 
                                                    (100, 100, 0))
            _, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
            _, A_pre_fp_exp = convert_to_fp(A_pre, 8)
            p_pre_fp_man, p_pre_fp_exp = convert_to_fp(p_pre, 24)
            correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp)
            self.ProjGrad = ProjectedGradientNeuronsPIPGeq(
                        shape=init_state.shape,
                        qp_neurons_init=init_state,
                        da_exp=correction_exp,
                        grad_bias=p_pre_fp_man,
                        grad_bias_exp=p_pre_fp_exp,
                        alpha=alpha_man,
                        alpha_exp=alpha_exp,
                        lr_decay_type="indices",
                        alpha_decay_params=decay_params,
                    )
            # Connect the parent InPort to the InPort of the Dense child-Process.
            proc.in_ports.a_in.connect(self.ProjGrad.in_ports.a_in_qc)
            self.ProjGrad.out_ports.s_out.connect(proc.out_ports.s_out_qc)
            proc.vars.variable_assignment.alias(self.ProjGrad.qp_neuron_state)
        else:
            AssertionError("Unknown neuron model specified")


@implements(proc=ContinuousConstraintsProcess, protocol=LoihiProtocol)
@requires(CPU)
class ContinuousConstraintsModel(AbstractSubProcessModel):
    """Model for the ContinuousConstraints process.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        neuron_model = proc.hyperparameters.get("neuron_model", "qp/lp-pipg")
        
        if neuron_model == "qp/lp-pipg":
            # initialize weight processes A and A^T
              # these are part of the problem not hyperparams
            # adding them here to show that they are need for the neurons models
            # since some values are calculated based on these weights 
            A_pre = proc.hyperparameters.get("A", 0)
            Q_pre = proc.hyperparameters.get("Q", 0)
            k_pre = proc.hyperparameters.get("k", 0)
            _, Q_pre_fp_exp = convert_to_fp(Q_pre, 8)
            A_pre_fp_man, A_pre_fp_exp = convert_to_fp(A_pre, 8)
            k_pre_fp_man, k_pre_fp_exp = convert_to_fp(k_pre, 24)
            correction_exp = min(A_pre_fp_exp, Q_pre_fp_exp)
            
            A_exp_new = -correction_exp + A_pre_fp_exp
            # legitimate hyperparameters
            init_constraints = proc.hyperparameters.get(
                                                    "init_constraints",
                                                  np.zeros((k_pre.shape[0],), 
                                                           dtype=int))
            beta_man =  proc.hyperparameters.get("beta_mantissa", 1)
            beta_exp =  proc.hyperparameters.get("beta_exponent", 1)
            growth_params = proc.hyperparameters.get("growth_schedule_parameters", (3, 2))
            
            
            self.sparse_A = Sparse(weights=A_pre_fp_man, num_message_bits=24,
            )
            
            self.sparse_A_T = Sparse(
                weights=A_pre_fp_man.T, weight_exp=A_exp_new, 
                num_message_bits=24
            )

            # Neurons for Constraint Checking
            self.ProInt = ProportionalIntegralNeuronsPIPGeq(
                        shape=init_constraints.shape,
                        constraint_neurons_init=init_constraints,
                        da_exp=A_pre_fp_exp,
                        thresholds=k_pre_fp_man,
                        thresholds_exp=k_pre_fp_exp,
                        beta=beta_man,
                        beta_exp=beta_exp,
                        lr_growth_type="indices",
                        beta_growth_params=growth_params,
                    )
            proc.in_ports.a_in.connect(self.sparse_A.s_in)
            self.sparse_A.a_out.connect(self.ProInt.a_in)
            self.ProInt.s_out.connect(self.sparse_A_T.s_in)
            self.sparse_A_T.a_out.connect(proc.out_ports.s_out)
            proc.vars.constraint_assignment.alias(self.ProInt.constraint_neuron_state)
        else:
            AssertionError("Unknown neuron model specified")
        

@implements(proc=DiscreteVariablesProcess, protocol=LoihiProtocol)
@requires(CPU)
class DiscreteVariablesModel(AbstractSubProcessModel):
    """Model for the DiscreteVariables process.

    The model composes a population of StochasticIntegrateAndFire units and
    connects them via Dense processes as to represent integer or binary
    variables.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        wta_weight = -2
        shape = proc.proc_params.get("shape", (1,))
        diagonal = proc.proc_params.get("cost_diagonal")
        weights = proc.proc_params.get(
            "weights",
            wta_weight
            * np.logical_not(np.eye(shape[1] if len(shape) == 2 else 0)),
        )
        neuron_model = proc.hyperparameters.get("neuron_model", "nebm")
        if neuron_model == "nebm":
            temperature = proc.hyperparameters.get("temperature", 1)
            refract = proc.hyperparameters.get("refract", 0)
            init_value = proc.hyperparameters.get("init_value",
                                                  np.zeros(shape, dtype=int))
            init_state = proc.hyperparameters.get("init_state",
                                                  np.zeros(shape, dtype=int))

            self.s_bit = NEBMAbstract(temperature=temperature,
                                      refract=refract,
                                      init_state=init_state,
                                      shape=shape,
                                      cost_diagonal=diagonal,
                                      init_value=init_value)
        elif neuron_model == 'scif':
            noise_amplitude = proc.hyperparameters.get("noise_amplitude", 1)
            noise_precision = proc.hyperparameters.get("noise_precision", 5)
            init_value = proc.hyperparameters.get("init_value", np.zeros(shape))
            init_state = proc.hyperparameters.get("init_value", np.zeros(shape))
            on_tau = proc.hyperparameters.get("sustained_on_tau", (-3))
            self.s_bit = \
                StochasticIntegrateAndFire(shape=shape,
                                           step_size=diagonal,
                                           init_state=init_state,
                                           init_value=init_value,
                                           noise_amplitude=noise_amplitude,
                                           noise_precision=noise_precision,
                                           sustained_on_tau=on_tau,
                                           cost_diagonal=diagonal)
        elif neuron_model == 'nebm-sa' or 'nebm-sa-balanced':
            max_temperature = proc.hyperparameters.get("max_temperature", 10)
            min_temperature = proc.hyperparameters.get("min_temperature", 0)
            delta_temperature = proc.hyperparameters.get("delta_temperature", 1)
            steps_per_temperature = proc.hyperparameters.get(
                "steps_per_temperature", 100)
            refract_scaling = proc.hyperparameters.get("refract_scaling", 14)
            refract = proc.hyperparameters.get("refract", 0)
            init_value = proc.hyperparameters.get("init_value",
                                                  np.zeros(shape, dtype=int))
            init_state = proc.hyperparameters.get("init_state",
                                                  np.zeros(shape, dtype=int))
            self.s_bit = \
                NEBMSimulatedAnnealingAbstract(
                    shape=shape,
                    max_temperature=max_temperature,
                    min_temperature=min_temperature,
                    delta_temperature=delta_temperature,
                    steps_per_temperature=steps_per_temperature,
                    refract=refract,
                    refract_scaling=refract_scaling,
                    init_value=init_value,
                    init_state=init_state,
                    neuron_model=neuron_model,
                )
        else:
            AssertionError("Unknown neuron model specified")
        if weights.shape != (0, 0):
            self.dense = Dense(weights=weights)
            self.s_bit.out_ports.messages.connect(self.dense.in_ports.s_in)
            self.dense.out_ports.a_out.connect(self.s_bit.in_ports.added_input)

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.a_in.connect(self.s_bit.in_ports.added_input)
        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.s_bit.out_ports.messages.connect(proc.out_ports.s_out)
        self.s_bit.out_ports.local_cost.connect(proc.out_ports.local_cost)
        proc.vars.variable_assignment.alias(self.s_bit.prev_assignment)


@implements(proc=CostConvergenceChecker, protocol=LoihiProtocol)
@requires(CPU)
class CostConvergenceCheckerModel(AbstractSubProcessModel):
    """Model for the CostConvergence process.

    The model composes a CostIntegrator unit with incomming connections,
    in this way, downstream processes can be directly connected to the
    CostConvergence process.
    """

    def __init__(self, proc):
        # Instantiate child processes
        # The input shape is a 2D vector (shape of the weight matrix).
        shape = proc.proc_params.get("shape", (1,))
        weights = proc.proc_params.get("weights", np.ones((1, shape[0])))
        self.dense = Dense(weights=weights, num_message_bits=24)
        self.cost_integrator = CostIntegrator(shape=(1,), min_cost=0)
        self.dense.out_ports.a_out.connect(
            self.cost_integrator.in_ports.cost_in
        )

        # Connect the parent InPort to the InPort of the Dense child-Process.
        proc.in_ports.cost_components.connect(self.dense.in_ports.s_in)

        # Connect the OutPort of the LIF child-Process to the OutPort of the
        # parent Process.
        self.cost_integrator.out_ports.update_buffer.connect(
            proc.out_ports.update_buffer
        )
        proc.vars.min_cost.alias(self.cost_integrator.vars.min_cost)
        proc.vars.cost.alias(self.cost_integrator.vars.cost)


@implements(proc=StochasticIntegrateAndFire, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class StochasticIntegrateAndFireModelSCIF(AbstractSubProcessModel):
    """Model for the StochasticIntegrateAndFire process.
    The process is just a wrapper over the QuboScif process.
    # Todo deprecate in favour of QuboScif.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        step_size = proc.proc_params.get("step_size", (1,))
        theta = proc.proc_params.get("threshold", (1,))
        cost_diagonal = proc.proc_params.get("cost_diagonal", (1,))
        noise_amplitude = proc.proc_params.get("noise_amplitude", (1,))
        noise_precision = proc.proc_params.get("noise_precision", (3,))
        sustained_on_tau = proc.proc_params.get("sustained_on_tau", (-5,))
        self.scif = QuboScif(
            shape=shape,
            cost_diag=cost_diagonal,
            theta=theta,
            sustained_on_tau=sustained_on_tau,
            noise_amplitude=noise_amplitude,
            noise_precision=noise_precision,
        )
        proc.in_ports.added_input.connect(self.scif.in_ports.a_in)
        self.scif.s_wta_out.connect(proc.out_ports.messages)
        self.scif.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.scif.vars.spk_hist)
        proc.vars.state.alias(self.scif.vars.state)
        proc.vars.cost_diagonal.alias(self.scif.vars.cost_diagonal)
        proc.vars.noise_amplitude.alias(self.scif.vars.noise_ampl)
        proc.vars.noise_precision.alias(self.scif.vars.noise_prec)


@implements(proc=NEBMAbstract, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NEBMAbstractModel(AbstractSubProcessModel):
    """Model for the StochasticIntegrateAndFire process.

    The process is just a wrapper over the Boltzmann process.
    # Todo deprecate in favour of Boltzmann.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        temperature = proc.proc_params.get("temperature", (1,))
        refract = proc.proc_params.get("refract", (1,))
        init_value = proc.proc_params.get("init_value", np.zeros(shape))
        init_state = proc.proc_params.get("init_state", np.zeros(shape))
        self.scif = NEBM(shape=shape,
                         temperature=temperature,
                         refract=refract,
                         init_value=init_value,
                         init_state=init_state)
        proc.in_ports.added_input.connect(self.scif.in_ports.a_in)
        self.scif.s_wta_out.connect(proc.out_ports.messages)
        self.scif.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.scif.vars.spk_hist)
        proc.vars.state.alias(self.scif.vars.state)


@implements(proc=NEBMSimulatedAnnealingAbstract, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NEBMSimulatedAnnealingAbstractModel(AbstractSubProcessModel):
    """Model for the StochasticIntegrateAndFire process.

    The process is just a wrapper over the Boltzmann process.
    # Todo deprecate in favour of Boltzmann.
    """

    def __init__(self, proc):
        shape = proc.proc_params.get("shape", (1,))
        max_temperature = proc.proc_params.get("max_temperature", 10)
        min_temperature = proc.proc_params.get("min_temperature", 0)
        delta_temperature = proc.proc_params.get("delta_temperature", 1)
        steps_per_temperature = proc.proc_params.get(
            "steps_per_temperature", 100)
        refract_scaling = proc.proc_params.get("refract_scaling", 14)
        refract = proc.proc_params.get("refract", (1,))
        init_value = proc.proc_params.get("init_value", np.zeros(shape))
        init_state = proc.proc_params.get("init_state", np.zeros(shape))
        neuron_model = proc.proc_params.get("neuron_model")
        self.scif = NEBMSimulatedAnnealing(
            shape=shape,
            max_temperature=max_temperature,
            min_temperature=min_temperature,
            delta_temperature=delta_temperature,
            steps_per_temperature=steps_per_temperature,
            refract_scaling=refract_scaling,
            refract=refract,
            init_value=init_value,
            init_state=init_state,
            neuron_model=neuron_model,
        )
        proc.in_ports.added_input.connect(self.scif.in_ports.a_in)
        self.scif.s_wta_out.connect(proc.out_ports.messages)
        self.scif.s_sig_out.connect(proc.out_ports.local_cost)

        proc.vars.prev_assignment.alias(self.scif.vars.spk_hist)
        proc.vars.state.alias(self.scif.vars.state)
