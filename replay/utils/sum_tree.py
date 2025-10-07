import numpy as np


class SumTree:
    """
    SumTree — a binary tree data structure used in **Prioritized Experience Replay (PER)**
    to efficiently sample transitions proportionally to their priority.

    Reference: *Schaul et al., "Prioritized Experience Replay", arXiv:1511.05952 (2015)*

    Each leaf node stores a priority value, and each internal node stores
    the sum of its children's priorities. This allows O(log N) time complexity
    for both priority updates and sampling by cumulative sum.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity (int): Maximum number of elements to store.
                            The tree will store `2 * capacity - 1` nodes internally.
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Recursively update parent nodes after a priority change."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find the index of the leaf node where cumulative sum crosses `s`."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # reached a leaf
            return idx

        if self.tree[left] == 0 and self.tree[right] == 0:
            return idx  # empty branches

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Returns the sum of all priorities."""
        return self.tree[0]

    def add(self, priority: float, data: object):
        """
        Add a new element to the tree with a given priority.

        If capacity is exceeded, the oldest element is overwritten (circular buffer).
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update the priority value of an existing element."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        """
        Sample a data point given a cumulative priority sum `s`.

        Args:
            s (float): A value in [0, total_priority). Used to select a sample.

        Returns:
            (idx, priority, data): Tuple containing:
                - idx (int): Index in the tree
                - priority (float): Priority value at that leaf
                - data (object): Stored experience
        """
        total = self.total()

        if self.n_entries == 0 or total == 0:
            raise ValueError("SumTree is empty — no samples available yet.")

        s = np.clip(s, 0, total - 1e-5)
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        # Handle unfilled or invalid indices
        if data_idx >= self.n_entries or self.data[data_idx] is None:
            data_idx = np.random.randint(0, self.n_entries)
            idx = data_idx + self.capacity - 1

        return idx, self.tree[idx], self.data[data_idx]
