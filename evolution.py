from player import Player
import numpy as np
from config import CONFIG
import copy
import math
from nn import min_random, max_random
from util import make_stat_files


class Evolution:

    def __init__(self, mode):
        self.mode = mode
        self.gen_num = 1
        self.mutation_count = 0
        self.crossover_count = 0

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        mutation_distribution = [0.2, 0.8]  # from each 10 values 1 value will change
        mutation_occurrence = [True, False]

        # mutating weights
        for w in child.nn.w[1:]:
            shape = w.shape
            mutations = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if np.random.choice(mutation_occurrence, 1, p=mutation_distribution)[0]:
                        mutations.append((i, j))

            for mutation in mutations:
                # make random number in [-1, 1)
                # print(f'mutation on : {mutation}, before : {w[mutation]}', end='')
                w[mutation] = np.random.random(size=1)[0] * (max_random - min_random) + min_random
                # print(f', after: {w[mutation]}')

        # mutating biases
        for b in child.nn.b[1:]:
            shape = b.shape
            mutations = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if np.random.choice(mutation_occurrence, 1, p=mutation_distribution)[0]:
                        mutations.append((i, j))

            for mutation in mutations:
                # make random number in [-1, 1)
                b[mutation] = np.random.random(size=1)[0] * (max_random - min_random) + min_random

    def crossover(self, parent1, parent2):
        child1_biases = [None]
        child2_biases = [None]

        child1_weights = [None]
        child2_weights = [None]

        for bias_index in range(1, len(parent1.nn.b)):
            shape = parent1.nn.b[bias_index].shape
            p1_bias = parent1.nn.b[bias_index].flatten()
            p2_bias = parent2.nn.b[bias_index].flatten()

            c1_bias = []
            c2_bias = []

            slice_index = math.floor(len(p1_bias) / 2)

            c1_bias.extend(p1_bias[:slice_index])
            c1_bias.extend(p2_bias[slice_index:])

            c2_bias.extend(p2_bias[:slice_index])
            c2_bias.extend(p1_bias[slice_index:])

            child1_biases.append(np.array(c1_bias).reshape(shape))
            child2_biases.append(np.array(c2_bias).reshape(shape))

        for weight_index in range(1, len(parent1.nn.w)):
            shape = parent1.nn.w[weight_index].shape
            p1_weight = parent1.nn.w[weight_index].flatten()
            p2_weight = parent2.nn.w[weight_index].flatten()

            c1_weight = []
            c2_weight = []

            slice_index = math.floor(len(p1_weight) / 2)

            c1_weight.extend(p1_weight[:slice_index])
            c1_weight.extend(p2_weight[slice_index:])

            c2_weight.extend(p2_weight[:slice_index])
            c2_weight.extend(p1_weight[slice_index:])

            child1_weights.append(np.array(c1_weight).reshape(shape))
            child2_weights.append(np.array(c2_weight).reshape(shape))

        return child1_biases, child1_weights, child2_biases, child2_weights

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            mutations_happens = 0
            crossover_happens = 0

            # num_players example: 150
            mutation_distribution = [0.9, 0.1]
            mutation_occurrence = [True, False]

            crossover_distribution = [0.6, 0.4]
            crossover_occurrence = [True, False]

            fitness_list = np.array([player.fitness for player in prev_players])
            fitness_list = fitness_list ** 1.3
            distribution = fitness_list / sum(fitness_list)

            candidates_indices = list(np.random.choice(range(0, len(prev_players)), num_players, p=distribution, replace=True))

            new_players = []
            while candidates_indices:
                assert len(candidates_indices) % 2 == 0, 'the num_players must be even number'
                parent1_index = candidates_indices.pop(0)
                parent2_index = candidates_indices.pop(0)

                parent1 = prev_players[parent1_index]
                parent2 = prev_players[parent2_index]

                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

                do_crossover = np.random.choice(crossover_occurrence, 1, p=crossover_distribution)[0]
                do_mutate_child1 = np.random.choice(mutation_occurrence, 1, p=mutation_distribution)[0]
                do_mutate_child2 = np.random.choice(mutation_occurrence, 1, p=mutation_distribution)[0]

                if do_crossover:
                    crossover_happens += 1
                    child1_biases, child1_weights, child2_biases, child2_weights = self.crossover(parent1, parent2)
                    child1.nn.set_biases(child1_biases)
                    child1.nn.set_weights(child1_weights)
                    child2.nn.set_biases(child2_biases)
                    child2.nn.set_weights(child2_weights)

                if do_mutate_child1:
                    mutations_happens += 1
                    self.mutate(child1)
                if do_mutate_child2:
                    mutations_happens += 1
                    self.mutate(child2)

                new_players.append(child1)
                new_players.append(child2)

            self.mutation_count = mutations_happens
            self.crossover_count = crossover_happens
            # print(f'mutations : {mutations_happens}, crossovers: {crossover_happens}')
            return new_players

    def next_population_selection(self, players, num_players, mode='QT'):
        # Q tournament
        selected_players = []
        selected_players_indices = []
        if mode == 'QT':
            tournament_size = 3
            for _ in range(num_players):
                tournament_candidates_indices = np.random.randint(0, high=len(players), size=tournament_size)
                tournament_candidates = [(i, players[i]) for i in tournament_candidates_indices]
                tournament_candidates.sort(key=lambda x: x[1].fitness, reverse=True)
                selected_players.append(copy.deepcopy(tournament_candidates[0][1]))
                selected_players_indices.append(tournament_candidates[0][0])

        # top-k
        else:
            candidates = [(i, players[i]) for i in range(len(players))]
            candidates.sort(key=lambda x: x[1].fitness, reverse=True)
            selected_players = [x[1] for x in candidates[:num_players]]
            selected_players_indices = [x[0] for x in candidates[:num_players]]

        make_stat_files(self.mode, self.gen_num, self.mutation_count, self.crossover_count, players, selected_players_indices)
        #
        self.gen_num += 1
        return selected_players
