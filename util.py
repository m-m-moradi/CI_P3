import os
from os.path import join, basename
import pickle
from pathlib import Path
import shutil
from config import CONFIG
import numpy as np
import json
import os
import datetime


def clear(mode):
    checkpoint_path = Path(join('checkpoint', mode))
    statistics_path = Path(join('statistics', mode))
    try:
        shutil.rmtree(checkpoint_path)
        shutil.rmtree(statistics_path)
    except OSError as e:
        pass

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    statistics_path.mkdir(parents=True, exist_ok=True)


def backup(mode):
    checkpoint_path = Path(join('checkpoint', mode))
    statistics_path = Path(join('statistics', mode))
    date = datetime.datetime.now().strftime("%b-%d-%Y-%H-%M-%S")

    shutil.make_archive(f'./backups/{date}_checkpoint', 'zip', f'./checkpoint/{mode}')
    shutil.make_archive(f'./backups/{date}_statistics', 'zip', f'./statistics/{mode}')


# save players of this generation in file
def save_generation(players, gen_num, mode):
    path = Path(join('checkpoint', mode, str(gen_num)))
    try:
        shutil.rmtree(path)
    except OSError as e:
        pass

    path.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(players):
        player_path = join(path, str(i))
        with open(player_path, 'wb') as file:
            pickle.dump(p, file)


# load players from file
def load_generation(checkpoint_path):
    files = os.listdir(checkpoint_path)
    prev_players = []
    for f in files:
        with open(join(checkpoint_path, f), 'rb') as file:
            p = pickle.load(file)
            prev_players.append(p)

    return prev_players


def make_stat_files(mode, gen_num, mutation_count, crossover_count, players, selected_players_indices):
    directory = f'./statistics/{mode}'
    file_name = f'generation_{gen_num}.json'
    data = {
        'generation_number': gen_num,
        'agents': {},
        'mutations': mutation_count,  # mutations happened to this agents
        'crossover': crossover_count  # crossovers happened to this agents
    }
    for index, agent in enumerate(players):
        data['agents'][index] = {}
        data['agents'][index]['score'] = agent.fitness
        data['agents'][index]['selected'] = True if index in selected_players_indices else False

    with open(f'{directory}/{file_name}', 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)

    output_file.close()


class Normalizer:
    def __init__(self, mode):
        self.velocity_range = None
        self.box_list_x_range = [0, CONFIG['WIDTH']]
        self.box_list_y_range = [0, CONFIG['HEIGHT']]
        self.agent_x_range = [0, CONFIG['WIDTH']]
        self.agent_y_range = [0, CONFIG['HEIGHT']]

        min_velocity = calculate_max_velocity(mode, falling=False)  # going up - speed is negative
        max_velocity = calculate_max_velocity(mode, falling=True)  # falling - speed is positive
        self.velocity_range = [min_velocity, max_velocity]

    def norm_speed(self, x):
        return np.interp(x, self.velocity_range, [0, 1])

    def norm_box_list_x(self, x):
        return np.interp(x, self.box_list_x_range, [0, 1])

    def norm_box_list_y(self, x):
        return np.interp(x, self.box_list_y_range, [0, 1])

    def norm_agent_x(self, x):
        return np.interp(x, self.agent_x_range, [0, 1])

    def norm_agent_y(self, x):
        return np.interp(x, self.agent_y_range, [0, 1])


# based on game physics in player.py
# some parameters are based on players parameters
def calculate_max_velocity(mode='gravity', falling=False):
    g = 9.8  # G constant
    v = 0  # initial velocity
    if not falling:  # going up
        pos = [0, CONFIG['HEIGHT']]
        direction = 1  # going up
        limit = 0
        while pos[1] > limit:
            if mode in ('gravity', 'helicopter'):
                v -= g * direction * (1 / 60)
                pos[1] += v
            else:
                v -= 6 * direction
                pos[1] += v * (1 / 40)
        return v

    else:
        pos = [0, 0]
        direction = -1  # falling
        limit = CONFIG['HEIGHT']
        while pos[1] < limit:
            if mode in ('gravity', 'helicopter'):
                v -= g * direction * (1 / 60)
                pos[1] += v
            else:
                v -= 6 * direction
                pos[1] += v * (1 / 40)
        return v

