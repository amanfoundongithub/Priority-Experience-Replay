from collections import Counter
import numpy as np

# Assume PrioritizedReplayBuffer and SumTree are defined as above
from ReplayBuffer.buffer import PriorityExperienceReplayBuffer

def test_prioritized_replay_buffer():
    buffer = PriorityExperienceReplayBuffer(capacity=4, alpha=0.6)

    transitions = [
        ("state_A", "action_A", 1),   # low reward
        ("state_B", "action_B", 10),  # medium reward
        ("state_C", "action_C", 20),  # higher reward
        ("state_D", "action_D", 30)   # highest reward
    ]

    # Add transitions with corresponding "reward" as priority
    for t in transitions:
        state, action, reward = t
        buffer.add((state, action, reward), priority=reward)

    print("\nBuffer Filled. Now Sampling...\n")

    # Sample 1000 times and count which transitions get picked
    sample_counter = Counter()
    for _ in range(1000):
        batch, idxs, weights = buffer.sample(batch_size=1, beta=0.4)
        key = batch[0][0]  # state string (e.g., 'state_B')
        sample_counter[key] += 1

    print("Sample Frequency:")
    for state in ["state_A", "state_B", "state_C", "state_D"]:
        print(f"{state}: {sample_counter[state]} times")

    # Single sample print
    batch, idxs, weights = buffer.sample(batch_size=2, beta=0.4)
    print("\nSampled Transitions:")
    for t, idx, w in zip(batch, idxs, weights):
        print(f"Transition: {t}, Tree Index: {idx}, IS Weight: {w:.4f}")
