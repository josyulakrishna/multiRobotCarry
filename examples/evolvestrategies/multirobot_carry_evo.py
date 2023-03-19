import tqdm as tqdm
from torch.multiprocessing import Pool
from torch.optim import Adam
from evostrat import compute_centered_ranks, MultivariateNormalPopulation
from evostrat import NormalPopulation
from MRCarry import MRCarry
import wandb
import pickle


if __name__ == '__main__':
    wandb.init(project="evo_multirobot_carry_local_passage", entity="josyula")
    """
    Lunar landers weights and biases are drawn from normal distributions with learned means and fixed standard deviation 0.1. This is the approach suggested in OpenAI ES [1]    

    [1] - Salimans, Tim, et al. "Evolution strategies as a scalable alternative to reinforcement learning." arXiv preprint arXiv:1703.03864 (2017).    
    """
    param_shapes = {k: v.shape for k, v in MRCarry().get_params().items()}
    population = NormalPopulation(param_shapes, MRCarry.from_params, device="cuda:0", std=0.01)

    learning_rate = 0.1
    iterations = 1000
    pop_size = 200

    optim = Adam(population.parameters(), lr=learning_rate)
    pbar = tqdm.tqdm(range(iterations))
    pool = Pool()

    for _ in pbar:
        optim.zero_grad()
        raw_fit = population.fitness_grads(pop_size, pool, compute_centered_ranks)
        optim.step()
        pbar.set_description("fit avg: %0.3f, std: %0.3f" % (raw_fit.mean().item(), raw_fit.std().item()))
        wandb.log({'raw_fit_mean': raw_fit.mean().item(), 'raw_fit_std':raw_fit.std().item()})
        if raw_fit.mean() > 1500:
            print("Solved.")
            state_params = dict(zip(param_shapes.keys(), population.parameters()))
            pickle.dump(state_params, open('/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/evolvestrategies/best_model.p', 'wb'))
            break

    pool.close()
