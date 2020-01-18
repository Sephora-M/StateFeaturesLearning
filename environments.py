from learning_maze import LearningMazeDomain
from lspi import domains, basis_functions, policy, lspi, solvers

def example_grid_maze():
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = [14, 13, 24, 23, 29, 28, 39, 38]  # range(height*width)
    walls_location = [50, 51, 52, 53, 54, 55, 56, 74, 75, 76, 77, 78, 79]
    obstacles_transition_probability = .2
    domain = domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=2000)

    return maze


def low_stretch_tree_maze(num_sample=100, length_sample=100):
    reward_location = [15]
    obstacles_location = []
    obstacles_transition_probability = .2
    domain = domains.SymmetricMazeDomain(rewards_locations=reward_location,
                                              obstacles_location=obstacles_location)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample, length_sample=length_sample)

    return maze


def tworooms(num_sample=100, length_sample=100, discount=0.9):
    height = 10
    width = 10
    reward_location = 18
    initial_state = None  # np.array([25])
    obstacles_location = []  # range(height*width)
    walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                      41, 42, 43, 44, 46, 47, 48, 49]
    obstacles_transition_probability = .2
    domain = domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample, length_sample=length_sample, discount=discount)

    return maze


def threerooms(num_sample=5000, length_sample=100):
    height = 50
    width = 100
    reward_location = 198
    initial_state = None  # np.array([25])
    obstacles_location = []  # range(height*width)
    walls_location = []
    walls_location.extend(range(100))
    walls_location.extend(range(4900, 5000))
    walls_location.extend(range(0, 5000, 100))
    walls_location.extend(range(99, 5000, 100))
    walls_location.extend(range(1600, 1670))
    walls_location.extend(range(1680, 1700))
    walls_location.extend(range(3200, 3220))
    walls_location.extend(range(3230, 3300))

    obstacles_transition_probability = .2
    domain = domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample, length_sample=length_sample)


    return maze


class FourRooms():

    def __init__(self, num_samples, length_sample, discount, initial_state=None):
        self.num_actions = 4
        self.height = 13
        self.width = 13
        self.initial_state = initial_state  # np.array([25])
        self.obstacles_location = []  # range(height*width)
        self.length_sample = length_sample
        self.num_samples = num_samples
        self.discount = discount
        self.sampling_policy = policy.Policy(basis_functions.FakeBasis(4), discount, 1)
        self.samples = []
        self.lspi_samples = []
        self.walks = []
        self.solver = solvers.LSTDQSolver()

        walls_location = []
        walls_location.extend(range(13))
        walls_location.extend(range(156, 169))
        walls_location.extend(range(0, 157, 13))
        walls_location.extend(range(12, 169, 13))
        walls_location.extend(range(78, 80))
        walls_location.extend(range(81, 85))
        walls_location.extend(range(97, 100))
        walls_location.extend(range(101, 104))
        walls_location.extend(range(6, 45, 13))
        walls_location.extend(range(58, 124, 13))
        walls_location.extend(range(149, 163, 13))
        self.walls_location = walls_location

        self.obstacles_transition_probability = .2

    def learn_n2v_basis(self, dimension, discount, epochs, window_size, learning_rate=0.5):
        basis = basis_functions.Node2vecBasis('',
            self.num_actions, self.domain.transition_probabilities, dimension, self.walks, epochs=epochs,
            window_size=window_size, workers=8)
        return basis

    def learn_basis(self, dimension, discount, epochs, window_size, learning_rate=0.5):
        basis = basis_functions.DiscountedNode2vecBasis(
            self.num_actions, self.domain.transition_probabilities, discount, dimension, self.walks, self.actions, epochs=epochs,
            window_size=window_size, learning_rate=learning_rate, workers=8)
        return basis

    def update_domain(self, new_reward_location, obstacles_location, num_samples, length_sample):
        self.reward_location = new_reward_location
        self.obstacles_location = obstacles_location
        self.num_samples = num_samples
        self.length_sample = length_sample
        self.domain = domains.GridMazeDomain(self.height, self.width, new_reward_location,
                                             self.walls_location, self.obstacles_location, self.initial_state,
                                             self.obstacles_transition_probability)
        self.compute_samples()

    def compute_samples(self, reset_samples=True, reset_policy=True, biased_walk=False):
        if reset_policy:
            self.sampling_policy = policy.Policy(basis_functions.FakeBasis(4), self.discount, 1)
        if reset_samples:
            self.samples = []
            self.lspi_samples = []
            self.walks = []
            self.actions = []
        for i in range(self.num_samples):
            if biased_walk:
                sample, walk, terminated, lspi_sample = self.domain.generate_unique_samples(self.length_sample,
                                                                                            self.sampling_policy)
            else:
                sample, walk, walk_actions, terminated, lspi_sample = self.domain.generate_samples(self.length_sample,
                                                                                     self.sampling_policy)
            self.samples.extend(sample)
            self.walks.append(walk)
            self.actions.append(walk_actions)
            # if terminated: # and len(self.lspi_samples) <= NUM_SAMPLES:
            self.lspi_samples.extend(lspi_sample)
        # print(self.lspi_samples)

    def run_lspi(self, basis, discount, max_iter, explore):
        basis_policy = policy.Policy(basis, discount, explore)
        learned_policy, distances, iterations = lspi.learn(self.lspi_samples, basis_policy, self.solver,
                                                           max_iterations=max_iter)

        return learned_policy, distances, iterations


def oneroom(plotV=True, num_sample=100, length_sample=100, computeV=False):
    height = 10
    width = 10
    reward_location = 9
    initial_state = None  # np.array([25])
    obstacles_location = []  # range(height*width)
    walls_location = []
    obstacles_transition_probability = .2
    domain = domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample, length_sample=length_sample)

    return maze


def obstacles_room(plotV=True, num_sample=100, length_sample=100, computeV=False):
    height = 10
    width = 10
    reward_location = 18
    initial_state = None  # np.array([25])
    obstacles_location = [12, 13, 22, 23,
                          35, 36, 45, 46,
                          62, 63, 72, 73,
                          67, 77]
    walls_location = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 20, 30, 40, 50, 60, 70, 80, 90,
                      9, 19, 29, 39, 49, 59, 69, 79, 89, 99,
                      90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    obstacles_transition_probability = .2
    domain = domains.GridMazeDomain(height, width, reward_location,
                                         walls_location, obstacles_location, initial_state,
                                         obstacles_transition_probability)
    maze = LearningMazeDomain(domain=domain, num_sample=num_sample, length_sample=length_sample)

    return maze