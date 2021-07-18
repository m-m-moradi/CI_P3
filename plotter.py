import shutil
import os
import json
import pprint
import matplotlib.pyplot as plt
import pathlib

archive = './backups/Jul-18-2021-02-59-14_statistics.zip'
directory = os.path.splitext(os.path.basename(archive))[0]
shutil.unpack_archive(archive, directory, format='zip')

generations_list = []
for filename in os.listdir(directory):
    input_file = open(f'{directory}/{filename}')
    data = json.load(input_file)
    generations_list.append(data)
    input_file.close()

shutil.rmtree(directory, ignore_errors=True)
generations_list.sort(key=lambda item: item['generation_number'])
pprint.pprint(generations_list, depth=2)

best_scores = []
worst_scores = []
mean_scores = []

agents = []
agents_indices = []
selected_agents = []
selected_agents_indices = []
not_selected_agents = []
not_selected_agents_indices = []


for generation_dict in generations_list:
    for index, key in enumerate(generation_dict['agents']):
        agent = generation_dict['agents'][key]
        agents.append(agent)
        if agent['selected']:
            selected_agents.append(agent)
            selected_agents_indices.append(index)

        else:
            not_selected_agents.append(agent)
            not_selected_agents_indices.append(index)

    scores = list([x['score'] for x in agents])
    selected_scores = list([x['score'] for x in selected_agents])
    not_selected_scores = list([x['score'] for x in not_selected_agents])

    best_scores.append(max(scores))
    worst_scores.append(min(scores))
    mean_scores.append(sum(scores) / len(scores))

    plt.figure(figsize=(16, 8))
    plt.scatter(not_selected_agents_indices, not_selected_scores, c='red', label='not selected')
    plt.scatter(selected_agents_indices, selected_scores, c='blue', label='selected')
    plt.legend()
    title = f'Generation {generation_dict["generation_number"]}'

    path = pathlib.Path(f'plots/{directory}')
    path.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.savefig(f'{path.as_posix()}/{title}.jpeg', dpi=100)
    plt.close()


plt.figure(figsize=(16, 8))
x_axis = range(1, len(generations_list) + 1)


plt.scatter(x_axis, best_scores, label='best scores', c='g')
plt.plot(x_axis, best_scores, '--', color='g', alpha=0.4)

plt.scatter(x_axis, worst_scores, label='worst scores', c='r')
plt.plot(x_axis, worst_scores, '--', color='r', alpha=0.4)

plt.scatter(x_axis, mean_scores, label='mean scores', c='b')
plt.plot(x_axis, mean_scores, '--', color='b', alpha=0.4)
plt.legend()
title = f'Game statistics'

path = pathlib.Path(f'plots/{directory}')
path.mkdir(parents=True, exist_ok=True)
plt.title(title)
plt.savefig(f'{path.as_posix()}/{title}.jpeg', dpi=300)
plt.close()






