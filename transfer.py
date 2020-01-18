from environments import FourRooms
import numpy as np
import pickle

K = 3
# X = [50, 100, 200, 300, 500]
# X = [20,30, 50, 70, 100]
# X = [1, 2, 5, 10, 20, 50, 100]
# X = [100, 200, 300, 500]
X = [5, 10, 20, 50, 100, 200, 300]
walk_length = 100
walk_lspi = 50
num_walks = 200
discount_basis = 0.8
discount_rl = 0.8
epochs = 2
window_size = 50
learning_rate = 0.5
MAX_LSPI_ITER = 200
MAX_STEPS = 200
d = 125
explore = 1


def simulate(num_states, reward_location, walls_location, domain, learned_policy, max_steps=100):
    learned_policy.explore = 0.
    all_steps_to_goal = {}
    all_samples = {}
    all_cumulative_rewards = {}
    mean_steps_to_goal = 0.
    mean_cumulative_rewards = 0.
    num_starting_states = 0
    for state in range(num_states):
        if state not in reward_location and state not in walls_location:
            num_starting_states += 1
            steps_to_goal = 0
            domain.reset(np.array([state]))
            absorb = False
            samples = []
            while (not absorb) and (steps_to_goal < max_steps):
                action = learned_policy.select_action(domain.current_state())
                sample = domain.apply_action(action)
                absorb = sample.absorb
                steps_to_goal += 1
                samples.append(sample)
            all_steps_to_goal[state] = steps_to_goal
            all_samples[state] = samples
            all_cumulative_rewards[state] = np.sum([s.reward for s in samples])
            mean_cumulative_rewards += all_cumulative_rewards[state]
            mean_steps_to_goal += steps_to_goal

    mean_cumulative_rewards /= num_starting_states
    mean_steps_to_goal /= num_starting_states

    return all_steps_to_goal, all_samples, all_cumulative_rewards, mean_steps_to_goal, mean_cumulative_rewards


def run_experiment(pkl_dir_name):
    # cases = {'corner': ([24], [])} #, 'corner_middle': [24, 144], 'corner2': [24, 42], 'four_objects': [24, 42, 144, 100]}
    cases = {'env1': ([70], []), 'env2': ([24], []), 'env3': ([24, 154], []), 'env4': ([24, 154], [100])}

    all_X_means_cumul_rewards = {}
    all_X_std_cumul_rewards = {}
    all_X_means_steps_to_goal = {}
    all_X_std_steps_to_goal = {}
    all_X_means_num_lspi_iter = {}
    all_X_std_num_lspi_iter = {}

    for case_name in cases.keys():
        all_X_means_cumul_rewards[case_name] = []
        all_X_std_cumul_rewards[case_name] = []
        all_X_means_steps_to_goal[case_name] = []
        all_X_std_steps_to_goal[case_name] = []
        all_X_means_num_lspi_iter[case_name] = []
        all_X_std_num_lspi_iter[case_name] = []

    for num_walks_lspi in X:
        print('x = ' + str(num_walks_lspi))
        x_means_cumul_rewards = {}
        x_mean_steps_to_goal = {}
        x_num_lspi_iter = {}
        for case_name in cases.keys():
            x_means_cumul_rewards[case_name] = []
            x_mean_steps_to_goal[case_name] = []
            x_num_lspi_iter[case_name] = []

        for k in range(K):
            print(str(k)+': building environement and learning basis...')
            environment = FourRooms(num_walks, walk_length, discount_rl,None)
            environment.update_domain([], [], num_walks, walk_length)
            basis = environment.learn_basis(d, discount_basis, epochs, window_size, learning_rate)

            for case_name, locations in cases.iteritems():
                print('\t learning weights of ' + case_name)
                environment.update_domain(locations[0], locations[1], num_walks_lspi, walk_lspi)
                learned_policy, distances, iterations = environment.run_lspi(basis, discount_rl, MAX_LSPI_ITER, 1)
                # mean_steps_to_goal, samples, mean_cumulative_rewards = \
                #     simulate1([144], environment.reward_location, environment.walls_location, environment.domain,
                #              learned_policy, max_steps=MAX_STEPS)
                all_steps_to_goal, all_samples, all_cumulative_rewards, mean_steps_to_goal, mean_cumulative_rewards = \
                simulate(169, environment.reward_location, environment.walls_location, environment.domain,
                          learned_policy, max_steps=MAX_STEPS)
                x_means_cumul_rewards[case_name].append(mean_cumulative_rewards)
                x_mean_steps_to_goal[case_name].append(mean_steps_to_goal)
                x_num_lspi_iter[case_name].append(iterations)
                print('\t mean_r = '+str(mean_cumulative_rewards))

        for case_name in cases.keys():
            all_X_means_cumul_rewards[case_name].append(np.mean(x_means_cumul_rewards[case_name]))
            # print('average reward for case %s = %.2f' % (case_name, np.mean(x_means_cumul_rewards[case_name])))
            all_X_std_cumul_rewards[case_name].append(np.std(x_means_cumul_rewards[case_name]))
            all_X_means_steps_to_goal[case_name].append(np.mean(x_mean_steps_to_goal[case_name]))
            all_X_std_steps_to_goal[case_name].append(np.std(x_mean_steps_to_goal[case_name]))
            all_X_means_num_lspi_iter[case_name].append(np.mean(x_num_lspi_iter[case_name]))
            all_X_std_num_lspi_iter[case_name].append(np.std(x_num_lspi_iter[case_name]))

    pkl = open(
        pkl_dir_name + '/' + 'fourRooms_' + '_wl' + str(walk_length) + '_nw' + str(num_walks) + '_discount' + str(
            discount_basis) + '_lr' + str(learning_rate) + '_epochs' + str(epochs) + '_ws' + str(
            window_size) + '_K' + str(K) +'_d' + str(d) + '_maxSteps' + str(MAX_STEPS) + '_discounted_XNWlspi2.2_Successors_s2v.pkl', 'wb')
    pickle.dump({'mean_steps': all_X_means_steps_to_goal, 'std_steps': all_X_std_steps_to_goal,
                 'mean_rewards': all_X_means_cumul_rewards, 'std_rewards': all_X_std_cumul_rewards,
                 'mean_num_lspi_iter': all_X_means_num_lspi_iter, 'std_num_lspi_iter': all_X_std_num_lspi_iter}, pkl)

    return all_X_means_cumul_rewards, all_X_std_cumul_rewards, all_X_means_steps_to_goal, all_X_std_steps_to_goal, all_X_means_num_lspi_iter, all_X_std_num_lspi_iter
