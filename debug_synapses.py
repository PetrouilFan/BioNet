import torch
import numpy as np
import matplotlib.pyplot as plt
from continues import RewardModulatedSTDPSynapse

def test_synapse_learning():
    """Test synapse learning with controlled input patterns"""
    print("Testing synapse learning mechanism...")
    
    # Create test synapse
    synapse = RewardModulatedSTDPSynapse(
        initial_weight=0.5,
        learning_rate=0.1  # Higher learning rate for testing
    )
    
    weights = []
    eligibility = []
    
    # Run test pattern
    for i in range(100):
        # Create spiking pattern
        pre_spike = 1.0 if i % 4 == 0 else 0.0
        post_spike = 1.0 if i % 4 == 2 else 0.0
        
        # Update traces
        synapse.update_traces(pre_spike, post_spike, i)
        
        # Apply reward every 10 steps
        if i % 10 == 0:
            reward = 1.0
            synapse.apply_reward(reward)
        
        # Record data
        weights.append(synapse.weight.item())
        eligibility.append(synapse.eligibility_trace)
        
        # Print updates
        if i % 10 == 0:
            print(f"Step {i}: Weight={synapse.weight.item():.4f}, Eligibility={synapse.eligibility_trace:.4f}")
    
    # Plot results
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
