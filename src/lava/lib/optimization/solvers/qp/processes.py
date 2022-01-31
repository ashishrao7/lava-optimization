# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
import numpy as np
import typing as ty


class ConstraintDirections(AbstractProcess):
    """Connections in the constraint-checking group of neurons.
    Realizes the following abstract behavior:
    a_out = weights * s_in

    intialize the constraintDirectionsProcess

        Kwargs
        ------
        shape : int tuple, optional
            Define the shape of the connections matrix as a tuple. Defaults to
            (1,1)
        constraint_directions : (1-D  or 2-D np.array), optional
            Define the directions of the linear constraint hyperplanes. This is
            'A' in the constraints of the QP. Defaults to 0
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        weights = kwargs.pop("constraint_directions", 0)
        self.weights = Var(shape=shape, init=weights)

        # Profiling
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)
        col_sum_init = np.count_nonzero(weights, axis=0)
        self.col_sum = Var(shape=col_sum_init.shape, init=col_sum_init)

class ConstraintNeurons(AbstractProcess):
    """Process to check the violation of the linear constraints of the QP. A
    graded spike corresponding to the violated constraint is sent from the out
    port.

    Realizes the following abstract behavior:
    s_out = (a_in - thresholds) * (a_in < thresholds)

    Intialize the constraintNeurons Process.

        Kwargs:
        ------
        shape : int tuple, optional
            Define the shape of the thresholds vector. Defaults to (1,1).
        thresholds : 1-D np.array, optional
            Define the thresholds of the neurons in the
            constraint checking layer. This is usually 'k' in the constraints
            of the QP. Default value of thresholds is 0.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.a_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        self.thresholds = Var(shape=shape, init=kwargs.pop("thresholds", 0))


class QuadraticConnectivity(AbstractProcess):
    """The connections that define the Hessian of the quadratic cost function
    Realizes the following abstract behavior:
    a_out = weights * s_in

    Intialize the quadraticConnectivity process.

        Kwargs:
        ------
        shape : int tuple, optional
            A tuple defining the shape of the connections matrix. Defaults to
            (1,1).
        hessian : 1-D  or 2-D np.array, optional
            Define the hessian matrix ('Q' in the cost function of the QP) in
            the QP. Defaults to 0.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.weights = Var(shape=shape, init=kwargs.pop("hessian", 0))


class SolutionNeurons(AbstractProcess):
    """The neurons that evolve according to the constraint-corrected gradient
    dynamics.
    Implements the abstract behaviour
    qp_neuron_state += (-alpha * (s_in_qc + grad_bias) - beta * s_in_cn)

    Intialize the solutionNeurons process.

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        qp_neurons_init : 1-D np.array, optional
            initial value of qp solution neurons
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the
            QP definition.
        alpha : 1-D np.array, optional
            Defines the learning rate for gradient descent. Defaults to 1.
        beta : 1-D np.array, optional
            Defines the learning rate for constraint-checking. Defaults to 1.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.a_in_qc = InPort(shape=(shape[0], 1))
        self.s_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.a_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint checking
        self.s_out_cc = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.ones((shape[0], 1)))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.ones((shape[0], 1)))
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.growth_counter = Var(shape=(1, 1), init=0)


class ConstraintNormals(AbstractProcess):
    """Connections influencing the gradient dynamics when constraints are
    violated.
    Realizes the following abstract behavior:
    a_out = weights * s_in

    Intialize the constraint normals to assign weights to constraint
    violation spikes.

        Kwargs:
        ------
        shape : int tuple, optional
            A tuple defining the shape of the connections matrix. Defaults to
            (1,1).
        constraint_normals : 1-D  or 2-D np.array
            Define the normals of the linear constraint hyperplanes. This is
            A^T in the constraints of the QP. Defaults to 0
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[1], 1))
        self.a_out = OutPort(shape=(shape[0], 1))
        self.weights = Var(
            shape=shape, init=kwargs.pop("constraint_normals", 0)
        )


class ConstraintCheck(AbstractProcess):
    """Check if linear constraints (equality/inequality) are violated for the
    qp. Recieves and sends graded spike from and to the gradientDynamics
    process. House the constraintDirections and constraintNeurons as
    sub-processes.

    Implements Abstract behavior:
    (constraint_matrix*x-constraint_bias)*(constraint_matrix*x<constraint_bias)

    Initialize constraintCheck Process.

        Kwargs:
        ------
        constraint_matrix : 1-D  or 2-D np.array, optional
        The value of the constraint matrix. This is 'A' in the linear
        constraints.
        constraint_bias : 1-D np.array, optional
            The value of the constraint bias. This is 'k' in the linear
            constraints.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        constraint_matrix = kwargs.pop("constraint_matrix", 0)
        shape = constraint_matrix.shape
        self.s_in = InPort(shape=(shape[1], 1))
        self.constraint_matrix = Var(shape=shape, init=constraint_matrix)
        self.constraint_bias = Var(
            shape=(shape[0], 1), init=kwargs.pop("constraint_bias", 0)
        )
        self.s_out = OutPort(shape=(shape[0], 1))


class GradientDynamics(AbstractProcess):
    """Perform gradient descent with constraint correction to converge at the
    solution of the QP.

    Implements Abstract behavior:
    -alpha*(Q@x_init + p)- beta*A_T@graded_constraint_spike

    Initialize gradientDynamics Process.

        Kwargs:
        ------
        hessian : 1-D  or 2-D np.array, optional
            Define the hessian matrix ('Q' in the cost function of the QP) in
            the QP. Defaults to 0.
        constraint_matrix_T : 1-D  or 2-D np.array, optional
            The value of the transpose of the constraint matrix. This is 'A^T'
            in the linear constraints.
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the QP
            definition.
        qp_neurons_init : 1-D np.array, optional
            Initial value of qp solution neurons
        alpha : 1-D np.array, optional
            Define the learning rate for gradient descent. Defaults to 1.
        beta : 1-D np.array, optional
            Define the learning rate for constraint-checking. Defaults to 1.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
    """

    def __init__(self, **kwargs: ty.Any):
        """ """
        super().__init__(**kwargs)
        hessian = kwargs.pop("hessian", 0)
        constraint_matrix_T = kwargs.pop("constraint_matrix_T", 0)
        shape_hess = hessian.shape
        shape_constraint_matrix_T = constraint_matrix_T.shape
        self.s_in = InPort(shape=(shape_constraint_matrix_T[1], 1))
        self.hessian = Var(shape=shape_hess, init=hessian)
        self.constraint_matrix_T = Var(
            shape=shape_constraint_matrix_T, init=constraint_matrix_T
        )
        self.grad_bias = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("grad_bias", np.zeros((shape_hess[0], 1))),
        )
        self.qp_neuron_state = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("qp_neurons_init", np.zeros((shape_hess[0], 1))),
        )
        self.alpha = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("alpha", np.ones((shape_hess[0], 1))),
        )
        self.beta = Var(
            shape=(shape_hess[0], 1),
            init=kwargs.pop("beta", np.ones((shape_hess[0], 1))),
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )

        self.s_out = OutPort(shape=(shape_hess[0], 1))


class ProjectedGradientNeuronsPIPGeq(AbstractProcess):
    """The neurons that evolve according to the projected gradient
    dynamics specified in the PIPG algorithm.
    Intialize the ProjectedGradientNeuronsPIPGeq process.
    Implements the abstract behaviour
        qp_neuron_state -= alpha*(a_in_qc + grad_bias + a_in_cn)

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        qp_neurons_init : 1-D np.array, optional
            initial value of qp solution neurons
        grad_bias : 1-D np.array, optional
            The bias of the gradient of the QP. This is the value 'p' in the
            QP definition.
        alpha : 1-D np.array, optional
            Defines the learning rate for gradient descent. Defaults to 1.
        lr_decay_type: string, optional
            Defines the nature of the learning rate, alpha's decay. "schedule"
            decays it for every alpha_decay_schedule timesteps. "indices" halves
            the learning rate for every timestep defined in alpha_decay_indices.
        alpha_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for alpha. Default intialization to a very high value
            of 10000.
        alpha_decay_indices: list, optional
            The iteration numbers at which value of alpha gets halved
            (right-shifted).
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        # In/outPorts that come from/go to the quadratic connectivity process
        self.a_in_qc = InPort(shape=(shape[0], 1))
        self.s_out_qc = OutPort(shape=(shape[0], 1))
        # In/outPorts that come from/go to the constraint normals process
        self.a_in_cn = InPort(shape=(shape[0], 1))
        # OutPort for constraint directions
        self.s_out_cd = OutPort(shape=(shape[0], 1))
        self.qp_neuron_state = Var(
            shape=shape, init=kwargs.pop("qp_neurons_init", np.zeros(shape))
        )
        self.grad_bias = Var(
            shape=shape, init=kwargs.pop("grad_bias", np.zeros(shape))
        )
        self.alpha = Var(
            shape=shape, init=kwargs.pop("alpha", np.ones((shape[0], 1)))
        )
        self.alpha_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("alpha_decay_schedule", 10000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.proc_params["alpha_decay_indices"] = kwargs.pop(
            "alpha_decay_indices", [10000]
        )
        self.proc_params["lr_decay_type"] = kwargs.pop(
            "lr_decay_type", "schedules"
        )


class ProportionalIntegralNeuronsPIPGeq(AbstractProcess):
    """The neurons that evolve according to the proportional integral
    dynamics specified in the PIPG algorithm.
    Implements the abstract behaviour.
        constraint_neuron_state += beta * (a_in - constraint_bias)
        s_out = constraint_neuron_state + beta * (a_in - constraint_bias)

    Intialize the ProportionalIntegralNeuronsPIPGeq process.

        Kwargs:
        -------
        shape : int tuple, optional
            A tuple defining the shape of the qp neurons. Defaults to (1,1).
        constraint_neurons_init : 1-D np.array, optional
            Initial value of constraint neurons
        thresholds : 1-D np.array, optional
            Define the thresholds of the neurons in the
            constraint checking layer. This is usually 'k' in the constraints
            of the QP. Default value of thresholds is 0.
        beta : 1-D np.array, optional
            Defines the learning rate for constraint-checking. Defaults to 1.
        lr_growth_type: string, optional
            Defines the nature of the learning rate, beta's growth. "schedule"
            grows it for every beta_growth_schedule timesteps. "indices" doubles
            the learning rate for every timestep defined in beta_growth_indices.
        beta_growth_schedule : int, optional
            The number of iterations after which one left shift operation takes
            place for beta. Default intialization to a very high value of
            10000.
        beta_growth_indices: list, optional
            The iteration numbers at which value of beta gets doubled
            (left-shifted).
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.a_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        self.constraint_neuron_state = Var(
            shape=shape,
            init=kwargs.pop("constraint_neurons_init", np.zeros(shape)),
        )
        self.constraint_bias = Var(
            shape=shape, init=kwargs.pop("thresholds", np.zeros(shape))
        )
        self.beta = Var(
            shape=shape, init=kwargs.pop("beta", np.ones((shape[0], 1)))
        )
        self.beta_growth_schedule = Var(
            shape=(1, 1), init=kwargs.pop("beta_growth_schedule", 10000)
        )
        self.growth_counter = Var(shape=(1, 1), init=0)
        self.proc_params["beta_growth_indices"] = kwargs.pop(
            "beta_growth_indices", [10000]
        )
        self.proc_params["lr_growth_type"] = kwargs.pop(
            "lr_growth_type", "schedules"
        )

class SigmaNeurons(AbstractProcess):
    """Process to accumate spikes into a state variable before being fed to
    another process.
    Realizes the following abstract behavior:
    a_out = self.x_internal + s_in
        Kwargs:
        ------
        shape : int tuple, optional
            Define the shape of the thresholds vector. Defaults to (1,1).
        x_sig_init : 1-D np.array, optional
            initial value of internal sigma neurons. Should be the same as 
            qp_neurons_init. Default value is 0.
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        # should be same as x_int_
        self.x_internal = Var(shape=shape, init=kwargs.pop("x_sig_init", 0))

        # Profiling Vars
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)

class DeltaNeurons(AbstractProcess):
    """Process to simulate Delta coding. A graded spike is sent only if the 
    difference delta for a neuron exceeds the spiking threshold, Theta
    Realizes the following abstract behavior:
    delta = np.abs(s_in - self.x_internal)
    s_out =  delta[delta > theta]  
        Kwargs:
        ------
        shape : int tuple, optional
            Define the shape of the thresholds vector. Defaults to (1,1).
        x_del_init : 1-D np.array, optional
            initial value of internal delta neurons. Should be the same as 
            qp_neurons_init. Default value is 0.
        theta : 1-D np.array, optional
            Defines the learning rate for gradient descent. Defaults to 1.
        theta_decay_type: string, optional
            Defines the nature of the learning rate, theta's decay. "schedule"
            decays it for every theta_decay_schedule timesteps. "indices" halves
            the learning rate for every timestep defined in alpha_decay_indices.
        theta_decay_schedule : int, optional
            The number of iterations after which one right shift operation
            takes place for theta. Default intialization to a very high value
            of 10000.
        theta_decay_indices: list, optional
            The iteration numbers at which value of theta gets halved
            (right-shifted).
    """

    def __init__(self, **kwargs: ty.Any):
        super().__init__(**kwargs)
        shape = kwargs.get("shape", (1, 1))
        self.s_in = InPort(shape=(shape[0], 1))
        self.s_out = OutPort(shape=(shape[0], 1))
        self.x_internal = Var(shape=shape, init=kwargs.pop("x_del_init", 0))
        self.theta = Var(
            shape=shape, init=kwargs.pop("theta", np.ones((shape[0], 1)))
        )
        self.theta_decay_schedule = Var(
            shape=(1, 1), init=kwargs.pop("theta_decay_schedule", 10000)
        )
        self.decay_counter = Var(shape=(1, 1), init=0)
        self.proc_params["theta_decay_indices"] = kwargs.pop(
            "theta_decay_indices", [10000]
        )
        self.proc_params["theta_decay_type"] = kwargs.pop(
            "theta_decay_type", "schedules"
        )
        
        # Profiling Vars
        self.synops = Var(shape=(1, 1), init=0)
        self.neurops = Var(shape=(1, 1), init=0)
        self.spikeops = Var(shape=(1, 1), init=0)