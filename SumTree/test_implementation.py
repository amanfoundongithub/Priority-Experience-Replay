from SumTree.implementation import SumTree
import numpy as np 


def test_sumtree_correctness():
    try:
        capacity = 4
        tree = SumTree(capacity)

        # Add known priorities and values
        values = ['A', 'B', 'C', 'D']
        priorities = [1.0, 2.0, 3.0, 4.0]

        for val, p in zip(values, priorities):
            tree.add(val, p)

        # Test total
        expected_total = sum(priorities)
        assert abs(tree.total() - expected_total) < 1e-6, "Total priority mismatch"

        # Test sampling
        counts = {k: 0 for k in values}
        n_samples = 10000
        for _ in range(n_samples):
            s = np.random.uniform(0, tree.total())
            _, _, data = tree.find(s)
            counts[data] += 1

        # Check roughly proportional sampling
        empirical_probs = [counts[v] / n_samples for v in values]
        expected_probs = [p / expected_total for p in priorities]

        for emp, exp in zip(empirical_probs, expected_probs):
            assert abs(emp - exp) < 0.03, f"Sampling error too large: expected {exp}, got {emp}"

        print("✅ SumTree Test Passed")

    except AssertionError as e:
        print(f"❌ SumTree Test Failed: {e}")