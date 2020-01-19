# StateFeaturesLearning

## Prerequisites 
numpy, gensim, pygsp

## lspi
The code in this folder was built on top of this [LSPI python package](https://pythonhosted.org/lspi-python/autodoc/modules.html]). 

Least Squares Policy Iteration (LSPI) implementation.

Implements the algorithms described in the paper

["Least-Squares Policy Iteration.” Lagoudakis, Michail G., and Ronald Parr. Journal of Machine Learning Research 4, 2003.](https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf)

The implementation is based on the [Matlab implementation](http://www.cs.duke.edu/research/AI/LSPI/) provided by the authors.

#### basis_funciton.py
This file implements the abstract class BasisFunction. A basis function is a function
that takes in a state vector and an action index and returns a vector of features. 

A few specific types of basis functions are further implemented in this file:
* __Fake Basis Funtion__: it simply ignores all inputs an returns a constant basis vector. 
Can be useful for random sampling.
* __One Dimensional Polynomial Basis Funcion__: simple polynomial features for a state with one dimension.
* __Radial Basis Function__: the Gaussian multidimensional radial basis function (RBF).
* __Proto-Value Basis Function__: the PVFs as described in [Mahadevan and Maggioni's work](http://www.jmlr.org/papers/volume8/mahadevan07a/mahadevan07a.pdf).
* __Node2Vec__: automatically learnt basis function using the [node2vec algorithm](https://dl.acm.org/citation.cfm?id=2939672.2939754).
* __DiscountedNode2vecBasis__:
* __
#### domains.py
Contains example domains that LSPI works on. In particular, it implements the __grid maze domain__ 
in which the state space is a set of nodes on a N1 by N2 grid. Most nodes are always accessible 
(= rooms with 1. transition probability), some
nodes might be inaccessible (= walls with 0. transition probability), and some
nodes might be difficult to access (= obstacles with p transition probability
0 < p < 1). There is one absorbing goal state that gives reward of 100;
all other states are non absorbing and do not give any reward.
#### lspi.py
Contains the main interface to LSPI algorithm.

#### node2vec.py
Implements the node2vec algorithm to learn node embeddings.

#### policy.py
Contains the LSPI policy class used for learning, executing and sampling policies.

#### sample.py
Contains the Sample class that respresents an LSPI sample tuple ``(s, a, r, s', absorb)`` : (state, action, observed reward, future state, absorbing state)

#### solvers.py
Implementation of LSTDQ solver with standard matrix solver (the algorithm from Figure 5 of the [LSPI paper]((https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf))) 

### learning_maze.py
This class implements maze environments such as the one depicted below.

![alt text](https://github.com/LASP-UCL/Graph-RL/blob/master/twowalls_maze.png)

In such environment, the green states are accessible rooms, the dark purple states are strict walls and the
yellow state is the goal state. An agent can be initially placed in any accessible state and it aims
at reaching the goal state.

The class implements methods for learning the PVF basis functions as well as polynomial, 
node2vec and state2vec basis functions. 

## environment.py
Helpers method to build grid environments (tworooms, fourrooms, etc.)

