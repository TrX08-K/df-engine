import numpy as np


class _Node:
    __slots__ = ("center", "half", "mass", "com", "children", "bodies")

    def __init__(self, center, half):
        self.center = center
        self.half = half
        self.mass = 0.0
        self.com = np.zeros(3, dtype=float)
        self.children = None
        self.bodies = []


class BarnesHutTree:
    """
    Barnes-Hut octree force evaluation for the DFE kernel.

    Uses the same softened point-mass force law and positive potential sum
    as DFEProductionKernel.compute_acceleration_and_potential. Distant cells
    are replaced by their total mass at their centre of mass (monopole
    approximation) when cell_width / distance < theta. With theta = 0 every
    cell is opened and the result equals direct summation.
    """

    def __init__(self, G_sim=6.67430e-11, epsilon=0.1, theta=0.5, leaf_capacity=1, max_depth=48):
        self.G = G_sim
        self.epsilon_sq = epsilon ** 2
        self.theta = theta
        self.leaf_capacity = leaf_capacity
        self.max_depth = max_depth
        self.root = None
        self.positions = None
        self.masses = None
        self.interaction_count = 0

    def build(self, positions, masses):
        self.positions = np.asarray(positions, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        lo = self.positions.min(axis=0)
        hi = self.positions.max(axis=0)
        center = 0.5 * (lo + hi)
        half = 0.5 * float(np.max(hi - lo)) * 1.0001 + 1e-12
        self.root = _Node(center, half)
        for i in range(len(self.masses)):
            self._insert(self.root, i, 0)
        self._summarise(self.root)
        return self

    def _child_index(self, node, pos):
        return (int(pos[0] > node.center[0])
                | (int(pos[1] > node.center[1]) << 1)
                | (int(pos[2] > node.center[2]) << 2))

    def _make_children(self, node):
        h = 0.5 * node.half
        node.children = []
        for k in range(8):
            offset = np.array([h if k & 1 else -h, h if k & 2 else -h, h if k & 4 else -h])
            node.children.append(_Node(node.center + offset, h))

    def _insert(self, node, i, depth):
        if node.children is None:
            node.bodies.append(i)
            if len(node.bodies) > self.leaf_capacity and depth < self.max_depth:
                self._make_children(node)
                bodies, node.bodies = node.bodies, []
                for b in bodies:
                    child = node.children[self._child_index(node, self.positions[b])]
                    self._insert(child, b, depth + 1)
            return
        child = node.children[self._child_index(node, self.positions[i])]
        self._insert(child, i, depth + 1)

    def _summarise(self, node):
        if node.children is None:
            if node.bodies:
                m = self.masses[node.bodies]
                node.mass = float(np.sum(m))
                if node.mass > 0.0:
                    node.com = (self.positions[node.bodies] * m[:, None]).sum(axis=0) / node.mass
                else:
                    node.com = self.positions[node.bodies].mean(axis=0)
            return
        total = 0.0
        weighted = np.zeros(3, dtype=float)
        for child in node.children:
            self._summarise(child)
            total += child.mass
            weighted += child.mass * child.com
        node.mass = total
        node.com = weighted / total if total > 0.0 else node.center.copy()

    def _contains(self, node, pos):
        return bool(np.all(np.abs(pos - node.center) <= node.half))

    def _add_term(self, target_pos, src_pos, m, accel, pot):
        r_vec = target_pos - src_pos
        dist_soft = np.sqrt(np.sum(r_vec ** 2) + self.epsilon_sq)
        self.interaction_count += 1
        accel += -self.G * m * r_vec / (dist_soft ** 3)
        return pot + m / dist_soft

    def compute_acceleration_and_potential(self, target_pos, target_idx):
        """
        Returns (acceleration, positive potential sum) at target_pos, omitting
        the body with index target_idx. Pass target_idx = -1 for a field point.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        accel = np.zeros(3, dtype=float)
        pot = 0.0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.mass == 0.0 and not node.bodies and node.children is None:
                continue
            if node.children is None:
                for b in node.bodies:
                    if b == target_idx:
                        continue
                    pot = self._add_term(target_pos, self.positions[b], self.masses[b], accel, pot)
                continue
            d = np.sqrt(np.sum((node.com - target_pos) ** 2))
            width = 2.0 * node.half
            if (not self._contains(node, target_pos)) and d > 0.0 and (width / d) < self.theta:
                pot = self._add_term(target_pos, node.com, node.mass, accel, pot)
            else:
                stack.extend(node.children)
        return accel, pot


def advance_subsystem_tree(positions, velocities, masses, dt, G_sim=6.67430e-11, epsilon=0.1, theta=0.5):
    """
    Semi-implicit Euler step identical in structure to
    DFEProductionKernel.advance_subsystem, with accelerations from a
    Barnes-Hut tree rebuilt at the start of the step.
    """
    tree = BarnesHutTree(G_sim=G_sim, epsilon=epsilon, theta=theta).build(positions, masses)
    next_v = np.copy(velocities)
    next_x = np.copy(positions)
    for i in range(len(masses)):
        accel, _ = tree.compute_acceleration_and_potential(tree.positions[i], i)
        next_v[i] = velocities[i] + accel * dt
    for i in range(len(masses)):
        next_x[i] = positions[i] + next_v[i] * dt
    return next_x, next_v
