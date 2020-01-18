from lspi import policy,basis_functions, solvers, lspi

import numpy as np

NUM_BASIS = 5
DEGREE = 3
DISCOUNT = .9
EXPLORE = 0
EPSILON_DECAY=.9
NUM_SAMPLES = 200
LEN_SAMPLE = 500
MAX_ITERATIONS = 100
MAX_STEPS = 500


class LearningMazeDomain():

    def __init__(self, domain, num_sample=NUM_SAMPLES, length_sample=LEN_SAMPLE, discount=DISCOUNT):

        self.domain = domain

        self.sampling_policy = policy.Policy(basis_functions.FakeBasis(4), discount, 1)

        self.num_samples = num_sample
        self.length_samples = length_sample
        self.samples = []
        self.lspi_samples = []
        self.walks = []
        self.discount = discount

        self.random_policy_cumulative_rewards = np.sum([sample.reward for
                                                        sample in self.samples])

        self.solver = solvers.LSTDQSolver()

    def compute_samples(self, reset_samples=True, reset_policy=False, biased_walk=False):
        if reset_policy:
            self.sampling_policy = policy.Policy(basis_functions.FakeBasis(4), self.discount, 1)
        if reset_samples:
            self.samples = []
            self.lspi_samples = []
            self.walks = []
            self.actions = []
        for i in range(self.num_samples):
            if biased_walk:
                sample, walk, terminated, lspi_sample = self.domain.generate_unique_samples(self.length_samples, self.sampling_policy)
            else:
                sample, walk, walk_actions,  terminated, lspi_sample = self.domain.generate_samples(self.length_samples, self.sampling_policy)
            self.samples.extend(sample)
            self.walks.append(walk)
            self.actions.append(walk_actions)
            # if terminated: # and len(self.lspi_samples) <= NUM_SAMPLES:
            self.lspi_samples.extend(lspi_sample)

    def learn_proto_values_basis(self, num_basis=NUM_BASIS, explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES,
                                 initial_policy=None, rpi_epochs=1, run_simulation=False):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.ProtoValueBasis(
                self.domain.learn_graph(self.samples), 4, num_basis), self.discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_polynomial_basis(self, degree=DEGREE, discount=DISCOUNT,
                               explore=EXPLORE, max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES,
                               initial_policy=None, run_simulation=False):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.OneDimensionalPolynomialBasis(degree, 4), discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_node2vec_basis(self, dimension=NUM_BASIS, walk_length=30, num_walks=10, window_size=10,
                             p=1, q=1, epochs=1, explore=EXPLORE, max_iterations=MAX_ITERATIONS,
                             max_steps=NUM_SAMPLES, initial_policy=None, edgelist ='node2vec/graph/NA.edgelist',
                             run_simulation=False, lspi_epochs=1):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.Node2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities,
                dimension=dimension, walks=self.walks, walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs), self.discount, explore)

        self.sampling_policy = initial_policy
        for i in range(lspi_epochs):
            learned_policy, distances = lspi.learn(self.lspi_samples, self.sampling_policy, self.solver,
                                               max_iterations=max_iterations)
            self.sampling_policy = learned_policy
            self.sampling_policy.explore *= EPSILON_DECAY
            self.compute_samples(True)
        # self.sampling_policy.explore = 1.
        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_discounted_node2vec_basis(self, dimension=NUM_BASIS, walk_length=30, num_walks=10, window_size=10, gamma=0.6,
                             p=1, q=1, epochs=1, learning_rate=0.5, explore=EXPLORE, max_iterations=MAX_ITERATIONS,
                             max_steps=NUM_SAMPLES, initial_policy=None, edgelist ='node2vec/graph/NA.edgelist',
                             run_simulation=False, lspi_epochs=1):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.DiscountedNode2vecBasis(
                edgelist, num_actions=4, transition_probabilities=self.domain.transition_probabilities, discount=self.discount,
                dimension=dimension, walks=self.walks, walk_length=walk_length, num_walks=num_walks, window_size=window_size,
                p=p, q=q, epochs=epochs, learning_rate=learning_rate), gamma, explore)

        self.sampling_policy = initial_policy
        for i in range(lspi_epochs):
            learned_policy, distances = lspi.learn(self.lspi_samples, self.sampling_policy, self.solver,
                                               max_iterations=max_iterations)
            self.sampling_policy = learned_policy
            self.sampling_policy.explore *= EPSILON_DECAY
            self.compute_samples(True)
        # self.sampling_policy.explore = 1.
        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_graphwave_basis(self, graph_edgelist, dimension, walk_length=30, num_walks=10, time_pts_range=[0, 25],
                              taus='auto', max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, nb_filters=1,
                              initial_policy=None, discount=DISCOUNT, explore=EXPLORE, run_simulation=False):

        # graph = self.domain.learn_graph(sample_length=walk_length, num_samples=num_walks,
        #                                 sampling_policy=self.sampling_policy)
        #
        # self.domain.write_edgelist(graph_edgelist, graph)

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.GraphWaveBasis(graph_edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             time_pts_range=time_pts_range, taus=taus,
                                                                             nb_filters=nb_filters), discount, explore)
        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_struc2vec_basis(self, dimension=30, walk_length=100, num_walks=50, window_size=10, epochs=1,
                              edgelist='node2vec/graph/tworooms.edgelist', max_iterations=MAX_ITERATIONS, discount=DISCOUNT,
                               explore=EXPLORE, max_steps=NUM_SAMPLES, initial_policy=None, run_simulation=False):

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.Struc2vecBasis(graph_edgelist=edgelist, num_actions=4,
                                                                             dimension=dimension,
                                                                             walk_length=walk_length,
                                                                             num_walks=num_walks,
                                                                             window_size=window_size, epochs=epochs)
                                         , discount, explore)

        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances

    def learn_gcn_basis(self, graph_edgelist, dimension, walk_length=30, num_walks=10, time_pts_range=[0, 25],
                              taus='auto', max_iterations=MAX_ITERATIONS, max_steps=NUM_SAMPLES, nb_filters=1,
                              initial_policy=None, discount=DISCOUNT, explore=EXPLORE, run_simulation=False, model_str='gcn_vae',):

        # graph = self.domain.learn_graph(sample_length=walk_length, num_samples=num_walks,
        #                                 sampling_policy=self.sampling_policy)
        #
        # self.domain.write_edgelist(graph_edgelist, graph)

        if initial_policy is None:
            initial_policy = policy.Policy(basis_functions.GCNBasis(graph_edgelist, num_actions=4,
                                                                             dimension=dimension, model_str='gcn_vae',), discount, explore)
        learned_policy, distances = lspi.learn(self.lspi_samples, initial_policy, self.solver,
                                               max_iterations=max_iterations)

        self.domain.reset()

        steps_to_goal = 0
        absorb = False
        samples = []

        if run_simulation:
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(self.domain.current_state())
                sample = self.domain.apply_action(action)
                absorb = sample.absorb
                if absorb:
                    print('Reached the goal in %d', steps_to_goal)
                steps_to_goal += 1
                samples.append(sample)

        return steps_to_goal, learned_policy, samples, distances