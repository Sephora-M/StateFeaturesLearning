# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc
import networkx as nx

from node2vec import node2vec
from struc2vec.src import struc2vec
from struc2vec.src import graph
from graphwave import graphwave
from gammanode2vec import gammanode2vec

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import scipy.sparse as sp

import time

from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from gae.optimizer import OptimizerAE, OptimizerVAE

import tensorflow as tf

import numpy as np


class BasisFunction(object):

    r"""ABC for basis functions used by LSPI Policies.

    A basis function is a function that takes in a state vector and an action
    index and returns a vector of features. The resulting feature vector is
    referred to as :math:`\phi` in the LSPI paper (pg 9 of the PDF referenced
    in this package's documentation). The :math:`\phi` vector is dotted with
    the weight vector of the Policy to calculate the Q-value.

    The dimensions of the state vector are usually smaller than the dimensions
    of the :math:`\phi` vector. However, the dimensions of the :math:`\phi`
    vector are usually much smaller than the dimensions of an exact
    representation of the state which leads to significant savings when
    computing and storing a policy.

    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).

        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix for the given state-action pair.

        The way this value is calculated depends entirely on the concrete
        implementation of BasisFunction.

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.


        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        """
        pass  # pragma: no cover

    @abc.abstractproperty
    def num_actions(self):
        """Return number of possible actions.

        Returns
        -------
        int
            Number of possible actions.
        """
        pass  # pragma: no cover

    @staticmethod
    def _validate_num_actions(num_actions):
        """Return num_actions if valid. Otherwise raise ValueError.

        Return
        ------
        int
            Number of possible actions.

        Raises
        ------
        ValueError
            If num_actions < 1

        """
        if num_actions < 1:
            raise ValueError('num_actions must be >= 1')
        return num_actions


class FakeBasis(BasisFunction):

    r"""Basis that ignores all input. Useful for random sampling.

    When creating a purely random Policy a basis function is still required.
    This basis function just returns a :math:`\phi` equal to [1.] for all
    inputs. It will however, still throw exceptions for impossible values like
    negative action indexes.

    """

    def __init__(self, num_actions):
        """Initialize FakeBasis."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

    def size(self):
        r"""Return size of 1.

        Returns
        -------
        int
            Size of :math:`phi` which is always 1 for FakeBasis

        Example
        -------

        >>> FakeBasis().size()
        1

        """
        return 1

    def evaluate(self, state, action):
        r"""Return :math:`\phi` equal to [1.].

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s. FakeBasis ignores these
            values.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a. FakeBasis ignores these
            values.

        Returns
        -------
        numpy.array
            :math:`\phi` vector equal to [1.].

        Raises
        ------
        IndexError
            If action index is < 0

        Example
        -------

        >>> FakeBasis().evaluate(np.arange(10), 0)
        array([ 1.])

        """
        if action < 0:
            raise IndexError('action index must be >= 0')
        if action >= self.num_actions:
            raise IndexError('action must be < num_actions')
        return np.array([1.])

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class OneDimensionalPolynomialBasis(BasisFunction):

    """Polynomial features for a state with one dimension.

    Takes the value of the state and constructs a vector proportional
    to the specified degree and number of actions. The polynomial is first
    constructed as [..., 1, value, value^2, ..., value^k, ...]
    where k is the degree. The rest of the vector is 0.

    Parameters
    ----------
    degree : int
        The polynomial degree.
    num_actions: int
        The total number of possible actions

    Raises
    ------
    ValueError
        If degree is less than 0
    ValueError
        If num_actions is less than 1

    """

    def __init__(self, degree, num_actions):
        """Initialize polynomial basis function."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        if degree < 0:
            raise ValueError('Degree must be >= 0')
        self.degree = degree

    def size(self):
        """Calculate the size of the basis function.

        The base size will be degree + 1. This basic matrix is then
        duplicated once for every action. Therefore the size is equal to
        (degree + 1) * number of actions


        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.


        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.size()
        6

        """
        return (self.degree + 1) * self.num_actions

    def evaluate(self, state, action):
        r"""Calculate :math:`\phi` matrix for given state action pair.

        The :math:`\phi` matrix is used to calculate the Q function for the
        given policy.

        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.

        Example
        -------

        >>> basis = OneDimensionalPolynomialBasis(2, 2)
        >>> basis.evaluate(np.array([2]), 0)
        array([ 1.,  2.,  4.,  0.,  0.,  0.])

        """
        if action < 0 or action >= self.num_actions:
            raise IndexError('Action index out of bounds')

        if state.shape != (1, ):
            raise ValueError('This class only supports one dimensional states')

        phi = np.zeros((self.size(), ))

        offset = (self.size()/self.num_actions)*action

        value = state[0]

        phi[offset:offset + self.degree + 1] = \
            np.array([pow(value, i) for i in range(self.degree+1)])

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class RadialBasisFunction(BasisFunction):

    r"""Gaussian Multidimensional Radial Basis Function (RBF).

    Given a set of k means :math:`(\mu_1 , \ldots, \mu_k)` produce a feature
    vector :math:`(1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
    e^{-\gamma || s - \mu_k ||^2})` where `s` is the state vector and
    :math:`\gamma` is a free parameter. This vector will be padded with
    0's on both sides proportional to the number of possible actions
    specified.

    Parameters
    ----------
    means: list(numpy.array)
        List of numpy arrays representing :math:`(\mu_1, \ldots, \mu_k)`.
        Each :math:`\mu` is a numpy array with dimensions matching the state
        vector this basis function will be used with. If the dimensions of each
        vector are not equal than an exception will be raised. If no means are
        specified then a ValueError will be raised
    gamma: float
        Free parameter which controls the size/spread of the Gaussian "bumps".
        This parameter is best selected via tuning through cross validation.
        gamma must be > 0.
    num_actions: int
        Number of actions. Must be in range [1, :math:`\infty`] otherwise
        an exception will be raised.

    Raises
    ------
    ValueError
        If means list is empty
    ValueError
        If dimensions of each mean vector do not match.
    ValueError
        If gamma is <= 0.
    ValueError
        If num_actions is less than 1.

    Note
    ----

    The numpy arrays specifying the means are not copied.

    """

    def __init__(self, means, gamma, num_actions):
        """Initialize RBF instance."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        if len(means) == 0:
            raise ValueError('You must specify at least one mean')

        if reduce(RadialBasisFunction.__check_mean_size, means) is None:
            raise ValueError('All mean vectors must have the same dimensions')

        self.means = means

        if gamma <= 0:
            raise ValueError('gamma must be > 0')

        self.gamma = gamma

    @staticmethod
    def __check_mean_size(left, right):
        """Apply f if the value is not None.

        This method is meant to be used with reduce. It will return either the
        right most numpy array or None if any of the array's had
        differing sizes. I wanted to use a Maybe monad here,
        but Python doesn't support that out of the box.

        Return
        ------
        None or numpy.array
            None values will propogate through the reduce automatically.

        """
        if left is None or right is None:
            return None
        else:
            if left.shape != right.shape:
                return None
        return right

    def size(self):
        r"""Calculate size of the :math:`\phi` matrix.

        The size is equal to the number of means + 1 times the number of
        number actions.

        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.

        """
        return (len(self.means) + 1) * self.num_actions

    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix.

        Matrix will have the following form:

        :math:`[\cdots, 1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
        e^{-\gamma || s - \mu_k ||^2}, \cdots]`

        where the matrix will be padded with 0's on either side depending
        on the specified action index and the number of possible actions.

        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.

        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.

        """
        if action < 0 or action >= self.num_actions:
            raise IndexError('Action index out of bounds')

        if state.shape != self.means[0].shape:
            raise ValueError('Dimensions of state must match '
                             'dimensions of means')

        phi = np.zeros((self.size(), ))
        offset = (len(self.means[0])+1)*action

        rbf = [RadialBasisFunction.__calc_basis_component(state,
                                                          mean,
                                                          self.gamma)
               for mean in self.means]
        phi[offset] = 1.
        phi[offset+1:offset+1+len(rbf)] = rbf

        return phi

    @staticmethod
    def __calc_basis_component(state, mean, gamma):
        mean_diff = state - mean
        return np.exp(-gamma*np.sum(mean_diff*mean_diff))

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            If value < 1.

        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class ExactBasis(BasisFunction):

    """Basis function with no functional approximation.

    This can only be used in domains with finite, discrete state-spaces. For
    example the Chain domain from the LSPI paper would work with this basis,
    but the inverted pendulum domain would not.

    Parameters
    ----------
    num_states: list
        A list containing integers representing the number of possible values
        for each state variable.
    num_actions: int
        Number of possible actions.
    """

    def __init__(self, num_states, num_actions):
        """Initialize ExactBasis."""
        if len(np.where(num_states <= 0)[0]) != 0:
            raise ValueError('num_states value\'s must be > 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self._num_states = num_states

        self._offsets = [1]
        for i in range(1, len(num_states)):
            self._offsets.append(self._offsets[-1]*num_states[i-1])

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return reduce(lambda x, y: x*y, self._num_states, 1)*self.__num_actions

    def get_state_action_index(self, state, action):
        """Return the non-zero index of the basis.

        Parameters
        ----------
        state: numpy.array
            The state to get the index for.
        action: int
            The state to get the index for.

        Returns
        -------
        int
            The non-zero index of the basis

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        """
        if action < 0:
            raise IndexError('action index must be >= 0')
        if action >= self.num_actions:
            raise IndexError('action must be < num_actions')

        base = action * int(self.size() / self.__num_actions)

        offset = 0
        for i, value in enumerate(state):
            offset += self._offsets[i] * state[i]

        return base + offset

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a single non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """
        if len(state) != len(self._num_states):
            raise ValueError('Number of state variables must match '
                             + 'size of num_states.')
        if len(np.where(state < 0)[0]) != 0:
            raise ValueError('state cannot contain negative values.')
        for state_var, num_state_values in zip(state, self._num_states):
            if state_var >= num_state_values:
                raise ValueError('state values must be <= corresponding '
                                 + 'num_states value.')

        phi = np.zeros(self.size())
        phi[self.get_state_action_index(state, action)] = 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class ProtoValueBasis(BasisFunction):

    """Proto-value basis functions.

    These basis functions are formed using the eigenvectors of the graph Laplacian
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.
    num_laplacian_eigenvectors
    """

    def __init__(self, graph, num_actions, num_laplacian_eigenvectors, lap_type='combinatorial'):
        """Initialize ExactBasis."""
        if graph is None:
            raise ValueError('graph cannot be None')

        if num_laplacian_eigenvectors < 1:
            raise ValueError('num_actions must be >= 1')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self.num_laplacian_eigenvectors = num_laplacian_eigenvectors

        self.graph = graph

        self.graph.compute_laplacian(lap_type=lap_type)
        self.graph.compute_fourier_basis(recompute=True)

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self.num_laplacian_eigenvectors * self.__num_actions

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self.num_laplacian_eigenvectors * self.__num_actions)

        action_window = action*self.num_laplacian_eigenvectors
        # for basis_fct in self.graph.U[state[0], 1:self.num_laplacian_eigenvectors + 1]:
        for basis_fct in self.graph.L[state[0], 1:self.num_laplacian_eigenvectors + 1].toarray()[0]:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class Node2vecBasis(BasisFunction):

    """Node2vec basis functions.

    These basis functions are learned using the node2vec algorithm
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.

    """

    def __init__(self, graph_edgelist, num_actions, transition_probabilities, dimension, walks,
                 walk_length=100, num_walks=50, window_size=10, p=1, q=1, epochs=1, workers=8):
        """Initialize ExactBasis."""
        if graph_edgelist is None:
            raise ValueError('graph cannot be None')

        if dimension < 0:
            raise ValueError('dimension must be >= 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension

        # self._nxgraph = self.read_graph(graph_edgelist)
        # self._walk_length = walk_length
        # self._num_walks = num_walks
        self._window_size = window_size
        # self._p = p
        # self._q = q
        self._epochs = epochs
        self._workers = workers
        self._mean = []

        # self.G = node2vec.Graph(self._nxgraph, False, self._p, self._q, transition_probabilities)
        # self.G.preprocess_transition_probs()
        # walks = self.G.simulate_random_walks(self._num_walks, self._walk_length)#, True, reward_locations)
        print(walks[:50])
        self.model = self.learn_embeddings(walks)

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension * self.__num_actions

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self._dimension*self.__num_actions)

        action_window = action*self._dimension

        try:
            basis_fcts = self.model[str(state[0])]
        except KeyError:
            basis_fcts = self._mean
            # basis_fcts = [0] * self._dimension

        for basis_fct in basis_fcts:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value

    def read_graph(self, edge_list):
        '''
        Reads the input network in networkx.
        '''

        G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = G.to_undirected()

        return G

    def learn_embeddings(self, walks):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=self._dimension, window=self._window_size, min_count=0, sg=1,
                         workers=self._workers, iter=self._epochs)
        self._mean = np.mean(model.wv[model.wv.vocab], axis=0)
        return model


class GraphWaveBasis(BasisFunction):

    """GraphWave basis functions.

    These basis functions are formed using structural node embeddings
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.
    num_laplacian_eigenvectors
    """

    def __init__(self, graph_edgelist, num_actions, dimension, time_pts_range=[0, 100], taus='auto', nb_filters=1):
        """Initialize ExactBasis."""
        if graph_edgelist is None:
            raise ValueError('graph cannot be None')

        if dimension < 1:
            raise ValueError('dimension must be >= 1')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension

        self._nxgraph = self.read_graph(graph_edgelist)

        chi, heat_print, taus = graphwave.graphwave_alg(self._nxgraph, np.linspace(time_pts_range[0],time_pts_range[1],
                                                        int(dimension/2)), taus=taus,
                                                        nb_filters=nb_filters, verbose=True)

        self.structural_emb = chi

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension * self.__num_actions

    def read_graph(self, edge_list):
        '''
        Reads the input network in networkx.
        '''

        G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = G.to_undirected()

        return G

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self._dimension * self.__num_actions)

        action_window = action*self._dimension
        for basis_fct in self.structural_emb[state[0]]:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class Struc2vecBasis(BasisFunction):

    """Node2vec basis functions.

    These basis functions are learned using the node2vec algorithm
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.

    """

    def __init__(self, graph_edgelist, num_actions, dimension,
                 walk_length=100, num_walks=50, window_size=10, epochs=1, workers=8, OPT1=False, OPT2=False):
        """Initialize ExactBasis."""
        if graph_edgelist is None:
            raise ValueError('graph cannot be None')

        if dimension < 0:
            raise ValueError('dimension must be >= 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension

        self._graph = graph.load_edgelist(graph_edgelist, undirected=True)
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._window_size = window_size
        self._epochs = epochs
        self._workers = workers
        self._OPT1 = OPT1
        self._OPT2 = OPT2

        self.G = struc2vec.Graph(self._graph, workers=self._workers)

        self.exec_struc2vec()
        self.model = self.learn_embeddings()

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension * self.__num_actions

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self._dimension*self.__num_actions)

        action_window = action*self._dimension

        try:
            basis_fcts = self.model[str(state[0])]
        except KeyError:
            basis_fcts = [0] * self._dimension

        for basis_fct in basis_fcts:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    def learn_embeddings(self):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = LineSentence('random_walks.txt')
        model = Word2Vec(walks, size=self._dimension, window=self._window_size, min_count=0, hs=1, sg=1,
                         workers=self._workers, iter=self._epochs)
        return model

    def exec_struc2vec(self):
        '''
        Pipeline for representational learning for all nodes in a graph.
        '''

        if self._OPT1:
            self.G.preprocess_neighbors_with_bfs_compact()
        else:
            self.G.preprocess_neighbors_with_bfs()

        if self._OPT2:
            self.G.create_vectors()
            self.G.calc_distances(compactDegree=self._OPT1)
        else:
            self.G.calc_distances_all_vertices(compactDegree=self._OPT1)

        self.G.create_distances_network()
        self.G.preprocess_parameters_random_walk()

        self.G.simulate_walks(self._num_walks, self._walk_length)


class GCNBasis(BasisFunction):

    """GraphWave basis functions.

    These basis functions are formed using structural node embeddings
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.
    num_laplacian_eigenvectors
    """

    def __init__(self, graph_edgelist, num_actions, dimension, learning_rate=0.01, epochs=300, hidden1=32, hidden2=16,
                 dropout=0., model_str='gcn_vae', use_features=0):

        """Initialize ExactBasis."""
        if graph_edgelist is None:
            raise ValueError('graph cannot be None')

        if dimension < 1:
            raise ValueError('dimension must be >= 1')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension

        adj, features = self.read_graph(graph_edgelist)

        # Store original adjacency matrix (without diagonal entries) for later
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        # adj = adj_train

        if use_features == 0:
            features = sp.identity(features.shape[0])  # featureless

        # Some preprocessing
        adj_norm = preprocess_graph(adj)

        # Define placeholders
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]

        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Create model
        model = None
        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, num_features, features_nonzero, hidden1, hidden2, dimension)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, hidden1, dimension)

        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm, learning_rate=learning_rate)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm, learning_rate=learning_rate)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        print("GCN Optimization Finished!")

        feed_dict.update({placeholders['dropout']: 0})
        self.embeddings = sess.run(model.z_mean, feed_dict=feed_dict)

    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension * self.__num_actions

    def read_graph(self, edge_list):
        '''
        Reads the input network in networkx.
        '''

        G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = G.to_undirected()

        adj = nx.adjacency_matrix(G)
        features = np.identity(adj.shape[0])

        # nx.draw_spectral(G, with_labels=True, node_color=node_colors)
        return adj, features

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self._dimension * self.__num_actions)

        action_window = action*self._dimension
        for basis_fct in self.embeddings[state[0]]:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value


class DiscountedNode2vecBasis(BasisFunction):

    """Node2vec basis functions.

    These basis functions are learned using the node2vec algorithm
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.

    """

    def __init__(self, num_actions, transition_probabilities, discount, dimension, walks, actions, epochs=1,
                 window_size=10, learning_rate=0.5, workers=8):
        """Initialize ExactBasis."""
        # if graph_edgelist is None:
        #     raise ValueError('graph cannot be None')

        if dimension < 0:
            raise ValueError('dimension must be >= 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension
        # print(walks[:50])
        # self._nxgraph = self.read_graph(graph_edgelist)
        # self._walk_length = walk_length
        # self._num_walks = num_walks
        self._window_size = window_size
        self._epochs = epochs
        self._workers = workers
        self._mean = []
        self._num_states = len(transition_probabilities)
        self._discount = discount
        self._learning_rate = learning_rate
        self.model, _ = self.learn_embeddings(walks,actions)


    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension * self.__num_actions

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        phi = np.zeros(self._dimension*self.__num_actions)

        action_window = action*self._dimension

        try:
            basis_fcts = self.model[str(state[0])]
        except KeyError:
            basis_fcts = self._mean
            # basis_fcts = [0] * self._dimension

        for basis_fct in basis_fcts:
            phi[action_window] = basis_fct
            action_window = action_window + 1

        return phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value

    def read_graph(self, edge_list):
        '''
        Reads the input network in networkx.
        '''

        G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = G.to_undirected()

        return G

    def learn_embeddings(self, walks, actions):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''

        model = gammanode2vec.DiscountedNode2Vec(self._num_states, self.num_actions, self._dimension, self._window_size,
                                                 walks, actions,
                                                 self._discount)
        embeddings, training_info = model.train_discounted_n2v(learning_rate=self._learning_rate,
                                                               num_epochs=self._epochs)
        self._mean = np.mean(np.array(embeddings.values()), axis=0)
        return embeddings, training_info

class DiscountedState2vecBasis(BasisFunction):

    """Node2vec basis functions.

    These basis functions are learned using the node2vec algorithm
    on an undirected graph formed from state transitions induced by the MDP

    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.

    """

    def __init__(self, num_actions, transition_probabilities, discount, dimension, walks, actions, epochs=1,
                 window_size=10, learning_rate=0.5, workers=8):
        """Initialize ExactBasis."""
        # if graph_edgelist is None:
        #     raise ValueError('graph cannot be None')

        if dimension < 0:
            raise ValueError('dimension must be >= 0')

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)

        self._dimension = dimension
        # print(walks[:50])
        # print(actions[:50])
        # self._nxgraph = self.read_graph(graph_edgelist)
        # self._walk_length = walk_length
        # self._num_walks = num_walks
        self._window_size = window_size
        self._epochs = epochs
        self._workers = workers
        self._mean = []
        self._num_states = len(transition_probabilities)
        self._discount = discount
        self._learning_rate = learning_rate
        self.model, _ = self.learn_embeddings(walks, actions)


    def size(self):
        r"""Return the vector size of the basis function.

        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return self._dimension # * self.__num_actions

    def evaluate(self, state, action):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.

        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.

        Returns
        -------
        numpy.array
            :math:`\phi` vector

        Raises
        ------
        IndexError
            If action index < 0 or action index > num_actions
        ValueError
            If the size of the state does not match the the size of the
            num_states list used during construction.
        ValueError
            If any of the state variables are < 0 or >= the corresponding
            value in the num_states list used during construction.
        """

        # phi = np.zeros(self._dimension)
        # phi = np.zeros(self._dimension*self.__num_actions)

        action_window = action*self._dimension

        try:
            basis_fcts = self.model[state[0], action]
        except KeyError:
            print("embedding not found for pair (%d,%d) : " % (state[0], action))
            basis_fcts = self._mean
            # basis_fcts = [0] * self._dimension

        # for basis_fct in basis_fcts:
        #     phi[action_window] = basis_fct
        #     action_window = action_window + 1

        return basis_fcts #phi

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.

        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.

        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value

    def read_graph(self, edge_list):
        '''
        Reads the input network in networkx.
        '''

        G = nx.read_edgelist(edge_list, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        G = G.to_undirected()

        return G

    def learn_embeddings(self, walks, actions):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''

        model = gammanode2vec.DiscountedNode2Vec(self._num_states, self.num_actions, self._dimension, self._window_size, walks, actions,
                                                 self._discount)
        embeddings, training_info = model.train_s2v(learning_rate=self._learning_rate,
                                                               num_epochs=self._epochs)
        self._mean = np.mean(np.array(embeddings.values()), axis=0)
        return embeddings, training_info