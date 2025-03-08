import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from continues import RewardModulatedSTDPSynapse, NeuralModule, ContinuouslyAdaptingNeuralSystem
from maze_environment import MazeEnvironment

class PerceptionModule(NeuralModule):
    def __init__(self, input_size=8, hidden_size=12):
        super(PerceptionModule, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=12,
            module_name="perception"
        )
        self.accepts_sensory = True

class MovementModule(NeuralModule):
    def __init__(self, input_size=12, hidden_size=16):
        super(MovementModule, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=4,
            module_name="movement"
        )
        self.is_motor = True

class MazeBrain:
    def __init__(self, env=None):
        self.system = ContinuouslyAdaptingNeuralSystem()
        self.env = env
        self.perception = PerceptionModule()
        self.movement = MovementModule()
        self.system.add_module("perception", self.perception)
        self.system.add_module("movement", self.movement)
        self.system.start()
        self.last_action = None
        self.action_cooldown = 0
        
    def process_observation(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        self.system.process_sensory_input(observation, "perception")
        action = self._get_next_action()
        return action
        
    def _get_next_action(self):
        time.sleep(0.05)
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
            return self.last_action if self.last_action is not None else 0
        try:
            module_name, output_spikes = self.system.action_queue.get_nowait()
            if module_name == "movement":
                if self.env is not None and hasattr(self.env, 'output_values'):
                    if isinstance(output_spikes, list) and len(output_spikes) == 4:
                        self.env.output_values = output_spikes.copy()
                if isinstance(output_spikes, list) and len(output_spikes) == 4:
                    if all(s == 0 for s in output_spikes):
                        action = random.randint(0, 3)
                    else:
                        action = output_spikes.index(max(output_spikes))
                    self.action_cooldown = 2
                    self.last_action = action
                    return action
        except Exception:
            pass
        return self.last_action if self.last_action is not None else 0
    
    def provide_reward(self, reward):
        if reward != 0:
            reward_float = float(reward)
            for module_name in ["perception", "movement"]:
                if module_name in self.system.modules_dict:
                    module = self.system.modules_dict[module_name]
                    for synapse in module.synapses.values():
                        synapse.apply_reward(reward_float)
    
    def stop(self):
        self.system.stop()

def test_maze_brain():
    print("Testing MazeBrain integration with MazeEnvironment...")
    env = MazeEnvironment(width=15, height=15, wall_density=0.2, cell_size=30)
    brain = MazeBrain(env=env)
    dummy_observation = [0]*8
    action = brain.process_observation(dummy_observation)
    print("Selected action:", action)
    brain.stop()
    env.close_pygame()

def test_synapse_learning():
    print("Testing synapse learning mechanism...")
    
    synapse = RewardModulatedSTDPSynapse(
        initial_weight=0.5,
        learning_rate=0.1
    )
    
    weights = []
    eligibility = []
    
    for i in range(100):
        pre_spike = 1.0 if i % 4 == 0 else 0.0
        post_spike = 1.0 if i % 4 == 2 else 0.0
        synapse.update_traces(pre_spike, post_spike, i)
        if i % 10 == 0:
            reward = 1.0
            synapse.apply_reward(reward)
        weights.append(synapse.weight.item())
        eligibility.append(synapse.eligibility_trace)
        if i % 10 == 0:
            print(f"Step {i}: Weight={synapse.weight.item():.4f}, Eligibility={synapse.eligibility_trace:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(weights)
    plt.title("Synapse Weight")
    plt.ylabel("Weight Value")
    plt.subplot(2, 1, 2)
    plt.plot(eligibility)
    plt.title("Eligibility Trace")
    plt.xlabel("Time Step")
    plt.ylabel("Eligibility Value")
    plt.tight_layout()
    plt.savefig("synapse_learning_test.png")
    plt.show()

if __name__ == "__main__":
    test_synapse_learning()
    test_maze_brain()
