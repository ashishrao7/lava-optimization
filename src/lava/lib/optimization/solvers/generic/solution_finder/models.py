# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.optimization.solvers.generic.dataclasses import (
    VariablesImplementation,
    CostMinimizer,
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    DiscreteVariablesProcess,
    ContinuousVariablesProcess,
    CostConvergenceChecker,
)
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.dense.process import Dense
from lava.lib.optimization.problems.variables import ContinuousVariables, \
    DiscreteVariables

@implements(proc=SolutionFinder, protocol=LoihiProtocol)
@requires(CPU)
class SolutionFinderModel(AbstractSubProcessModel):
    def __init__(self, proc):
        problem = proc.problem
        if not hasattr(problem, "variables"):
            raise Exception(
                "An optimization problem must contain " "variables."
            )
        if hasattr(problem.variables, "continuous") or isinstance(
            problem.variables, ContinuousVariables
        ):
           continuous_var_shape=problem.variables.continuous.num_variables

        if hasattr(problem.variables, "discrete") or isinstance(
            problem.variables, DiscreteVariables
        ):
            discrete_var_shape=problem.variables.discrete.num_variables,
            
        self.cost_diagonal = None

        if hasattr(problem, "cost"):
            cost_coefficients = problem.cost.coefficients
            cost_diagonal = problem.cost.coefficients[2].diagonal()

        hyperparameters = proc.proc_params.get("hyperparameters")

        # Subprocesses
        self.variables = VariablesImplementation()
        if discrete_var_shape:
            hyperparameters.update(
                dict(
                    init_state=self._get_init_state(
                        hyperparameters, cost_coefficients, discrete_var_shape
                    )
                )
            )
            self.variables.discrete = DiscreteVariablesProcess(
                shape=discrete_var_shape,
                cost_diagonal=cost_diagonal,
                hyperparameters=hyperparameters,
            )

        if continuous_var_shape:
            self.variables.continuous = ContinuousVariablesProcess(
                shape=continuous_var_shape
            )

        self.cost_minimizer = None
        self.cost_convergence_check = None
        if cost_coefficients is not None:
            self.cost_minimizer = CostMinimizer(
                Dense(
                    # todo just using the last coefficient for now
                    weights=cost_coefficients[2].init,
                    num_message_bits=24,
                )
            )
            self.variables.importances = cost_coefficients[1].init
            self.cost_convergence_check = CostConvergenceChecker(
                shape=discrete_var_shape
            )

        # Connect processes
        self.cost_minimizer.gradient_out.connect(self.variables.gradient_in)
        self.variables.state_out.connect(self.cost_minimizer.state_in)
        self.variables.local_cost.connect(
            self.cost_convergence_check.cost_components
        )

        proc.vars.variables_assignment.alias(
            self.variables.variables_assignment
        )
        proc.vars.cost.alias(
            self.cost_convergence_check.cost
        )
        self.cost_convergence_check.update_buffer.connect(
            proc.out_ports.cost_out
        )

    def _get_init_state(
        self, hyperparameters, cost_coefficients, discrete_var_shape
    ):
        init_value = hyperparameters.get(
            "init_value", np.zeros(discrete_var_shape, dtype=int)
        )
        
        q_off_diag =  cost_coefficients[2] * np.logical_not(
                np.eye(*cost_coefficients[2].shape)
            )
        q_diag = cost_coefficients[1]
        return q_off_diag @ init_value + q_diag
