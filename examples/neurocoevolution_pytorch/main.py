from pathlib import Path

import yaml
# from algorithms.trainer_ga import GATrainer
from algorithms.trainer_es import ESTrainer
config = {}
import torch.multiprocessing as mp
# import wandb
# wandb.init(project="neurocoevolution_mr_carry_nes", entity="josyula")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
plt.show()
mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    with open('configs/config_ga_test.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # trainer = GATrainer(config)
    trainer = ESTrainer(config)
    rewards = []
    p1 = []
    p2 = []
    while trainer.generation < config['stop_criteria']['generations']:
        summary = trainer.step_hof()
        print(summary)
        p1.append(summary['player1_fit'])
        p2.append(summary['player2_fit'])
        plt.plot(list(range(len(p1))) , p1, 'r-')
        plt.plot(list(range(len(p2))) , p2, 'b-')
        rewards.append((p1[-1]+p2[-1])/2.)
        plt.plot(list(range(len(rewards))), rewards, 'g-')
        plt.pause(0.05)
        # wandb.log(trainer.step())