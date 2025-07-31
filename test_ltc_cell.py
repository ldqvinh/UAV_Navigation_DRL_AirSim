# test_ltc_cell.py
import torch
from ncps.torch import LTC
import matplotlib.pyplot as plt

def test_ltc_dynamics():
    """
    Tests the dynamic properties of the LTC cell to ensure stability.
    """
    print("\n--- Task 2: Testing LTC Cell Dynamics ---")
    
    # We use a simple 1-input, 1-hidden-unit cell to make dynamics easy to visualize
    input_size = 1
    hidden_size = 1
    ltc_cell = LTC(input_size, hidden_size)

    # --- Test 1: Decay with Zero Input ---
    print("\n[Test 1] Verifying state decay with zero input...")
    
    # We'll simulate 50 timesteps
    sequence_length = 50
    
    # Input is all zeros
    zero_input = torch.zeros(sequence_length, 1, input_size) # Shape: (50, 1, 1)
    
    # Start with a non-zero hidden state to observe the decay
    # Shape: (batch_size, hidden_size) -> (1, 1)
    hidden_state = torch.ones(1, hidden_size) 
    
    # Store the hidden state at each timestep for plotting
    decay_outputs = []
    for t in range(sequence_length):
        # We process one timestep at a time, simulating a real-time rollout
        # Input shape for one step: (1, 1, 1) -> (sequence=1, batch=1, features=1)
        _, hidden_state = ltc_cell(zero_input[t:t+1], hx=hidden_state)
        decay_outputs.append(hidden_state.item())
    
    # Verification: The final state must be closer to zero than the initial state
    assert abs(decay_outputs[-1]) < abs(decay_outputs[0]), "FAIL: Hidden state did not decay."
    print(f"  Initial state: {decay_outputs[0]:.4f}, Final state: {decay_outputs[-1]:.4f}. ✅ Decay verified.")
    
    # --- Test 2: Convergence with Constant Positive Input ---
    print("\n[Test 2] Verifying state convergence with constant input...")
    
    # Input is a constant positive value
    const_input = torch.ones(sequence_length, 1, input_size) * 2.0
    
    # Start from a zero hidden state
    hidden_state = torch.zeros(1, hidden_size)
    
    convergence_outputs = []
    for t in range(sequence_length):
        _, hidden_state = ltc_cell(const_input[t:t+1], hx=hidden_state)
        convergence_outputs.append(hidden_state.item())
        
    # Verification: The state should stabilize, meaning the change in the last few steps is very small
    change_in_last_steps = abs(convergence_outputs[-1] - convergence_outputs[-5])
    assert change_in_last_steps < 1e-3, "FAIL: Hidden state did not converge."
    print(f"  Final state: {convergence_outputs[-1]:.4f}. ✅ Convergence verified.")

    # --- Visualization ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(decay_outputs)
    plt.title("Test 1: Hidden State Decay (Zero Input)")
    plt.xlabel("Timestep")
    plt.ylabel("Hidden State Value")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(convergence_outputs)
    plt.title("Test 2: Hidden State Convergence (Constant Input)")
    plt.xlabel("Timestep")
    plt.ylabel("Hidden State Value")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("ltc_cell_dynamics.png")
    print("\nSaved dynamics plot to ltc_cell_dynamics.png")
    print("\n--- Task 2 Complete ---")

if __name__ == "__main__":
    # Ensure you have matplotlib installed: pip install matplotlib
    test_ltc_dynamics()