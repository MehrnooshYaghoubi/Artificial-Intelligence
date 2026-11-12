"""
8-Puzzle Problem
====================================================

This code defines the 8-puzzle state space and the structure
of a general search algorithm. Students must implement the
search strategy (BFS, DFS, A*) by completing the designated
sections below.

💡 Encouragement for creativity:
   You are welcome (and encouraged!) to implement your own
   search strategies.
   See the "EXTENDING WITH NEW STRATEGIES" section below.
"""

import copy
import heapq
from typing import List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time

# ------------------------------------------------------------
# State Representation
# ------------------------------------------------------------


class PuzzleState:
    """
    Represents a state in the 8-puzzle.
    State is represented as a list of 9 integers (0 = blank tile).
    Example goal: [1, 2, 3, 4, 5, 6, 7, 8, 0]
    """

    def __init__(self, tiles: List[int], parent=None, action=None, cost=0):
        self.tiles = tiles
        self.parent = parent
        self.action = action
        self.cost = cost

    def __eq__(self, other):
        return isinstance(other, PuzzleState) and self.tiles == other.tiles

    def __hash__(self):
        return hash(tuple(self.tiles))

    def __str__(self):
        """Pretty-print 3x3 layout."""
        return "\n".join(
            [f"{self.tiles[0:3]}", f"{self.tiles[3:6]}", f"{self.tiles[6:9]}"]
        )

    def __lt__(self, other):
        """For heap comparisons in A* - compare by f(n) values"""
        return (self.cost + heuristic(self)) < (other.cost + heuristic(other))


# ------------------------------------------------------------
# Initial States (by Difficulty) and Goal State
# ------------------------------------------------------------

EASY_INITIAL_STATE = PuzzleState([1, 2, 3, 4, 0, 6, 7, 5, 8])

MEDIUM_INITIAL_STATE = PuzzleState([1, 3, 6, 5, 0, 2, 4, 7, 8])

HARD_INITIAL_STATE = PuzzleState([7, 2, 4, 5, 0, 6, 8, 3, 1])

GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]

# ------------------------------------------------------------
# Actions and Helper Functions
# ------------------------------------------------------------

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Pre-compute goal positions for heuristic calculation
GOAL_POSITIONS = {}
for i, tile in enumerate(GOAL_STATE):
    if tile != 0:
        GOAL_POSITIONS[tile] = divmod(i, 3)

GOAL_BLANK_ROW, GOAL_BLANK_COL = divmod(GOAL_STATE.index(0), 3)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def blank_context(state: "PuzzleState") -> Tuple[int, int]:
    """
    Bucket a board by where the blank sits relative to its goal position:
    (row_sign, col_sign), each in {-1, 0, 1}.

    An action like "UP" does not have one fixed value — moving the blank up
    helps when the blank is below its goal, and hurts when it's above it.
    Keying bandit statistics by this context (instead of by action alone)
    is what makes the stats mean something: each context is a genuinely
    different bandit problem, not noise averaged over unrelated situations.
    """
    row, col = divmod(state.tiles.index(0), 3)
    return (_sign(row - GOAL_BLANK_ROW), _sign(col - GOAL_BLANK_COL))


def successor_function(state: PuzzleState) -> List[Tuple[str, PuzzleState]]:
    """Return list of (action, next_state) pairs reachable from the given state."""
    successors = []
    index = state.tiles.index(0)
    row, col = divmod(index, 3)

    def swap_and_create(new_row, new_col, action):
        new_tiles = copy.deepcopy(state.tiles)
        new_index = new_row * 3 + new_col
        new_tiles[index], new_tiles[new_index] = new_tiles[new_index], new_tiles[index]
        return (
            action,
            PuzzleState(
                new_tiles,
                parent=state,
                action=action,
                cost=state.cost + cost_function(state, action, new_tiles),
            ),
        )

    if row > 0:
        successors.append(swap_and_create(row - 1, col, "UP"))
    if row < 2:
        successors.append(swap_and_create(row + 1, col, "DOWN"))
    if col > 0:
        successors.append(swap_and_create(row, col - 1, "LEFT"))
    if col < 2:
        successors.append(swap_and_create(row, col + 1, "RIGHT"))

    return successors


def cost_function(
    current_state: PuzzleState, action: str, next_state_tiles: List[int]
) -> int:
    """Uniform step cost of 1 for all actions."""
    return 1


def goal_test(state: PuzzleState) -> bool:
    """Check if the current state is the goal."""
    return state.tiles == GOAL_STATE


# ------------------------------------------------------------
# Heuristic Function for A* (and other informed strategies)
# ------------------------------------------------------------


def heuristic_chebyshev(state: PuzzleState) -> int:
    """
    IMPLEMENTATION of the Chebyshev Heuristic:
    2 * max(horizontal distance, vertical distance) summed over all tiles.
    This heuristic is more informed than the Manhattan distance.
    """
    h_c = 0

    # Iterate through the 9 tiles (indices 0 to 8)
    for i, tile in enumerate(state.tiles):
        # Skip the blank tile (0) as its position doesn't affect the goal state
        if tile == 0:
            continue

        # Get the tile's current position (r_current, c_current)
        r_current, c_current = divmod(i, 3)

        # Get the tile's goal position (r_goal, c_goal)
        r_goal, c_goal = GOAL_POSITIONS[tile]

        # Calculate horizontal and vertical distances
        delta_x = abs(c_current - c_goal)
        delta_y = abs(r_current - r_goal)

        # Calculate the Chebyshev distance: max(dx, dy)
        chebyshev_distance = max(delta_x, delta_y)

        # Calculate the Chebyshev heuristic contribution: 2 * chebyshev_distance
        # The paper uses 2*Chebyshev distance as the heuristic
        h_c += 2 * chebyshev_distance

    return h_c


def heuristic_manhattan(state: PuzzleState) -> int:
    """
    Manhattan Distance Heuristic:
    Sum of the horizontal and vertical distances of each tile from its goal position.
    Also known as "taxicab distance" or "L1 distance".

    This is admissible because it never overestimates the cost to reach the goal.
    Each tile must move at least its Manhattan distance to reach its goal position.
    """
    h_m = 0

    # Iterate through the 9 tiles (indices 0 to 8)
    for i, tile in enumerate(state.tiles):
        # Skip the blank tile (0) as its position doesn't affect the goal
        if tile == 0:
            continue

        # Get the tile's current position (row, column)
        r_current, c_current = divmod(i, 3)

        # Get the tile's goal position (row, column)
        r_goal, c_goal = GOAL_POSITIONS[tile]

        # Calculate Manhattan distance: |x1 - x2| + |y1 - y2|
        manhattan_distance = abs(r_current - r_goal) + abs(c_current - c_goal)

        # Add to total heuristic value
        h_m += manhattan_distance

    return h_m


def heuristic(state: PuzzleState) -> int:
    return heuristic_chebyshev(state)


# ------------------------------------------------------------
# 🧠 CREATIVE STRATEGY: Memory-Guided A*
# ------------------------------------------------------------


def memory_guided_astar(initial_state: PuzzleState) -> Optional[PuzzleState]:
    """
    MEMORY-GUIDED A* (contextual-bandit tie-breaking)

    A twist on A* with balanced exploration-exploitation!

    The idea: track which actions (UP/DOWN/LEFT/RIGHT) successfully reduce
    the heuristic value, using an "optimism bonus" for under-explored
    actions (UCB, as in multi-armed bandits / AlphaGo) -- BUT keep separate
    statistics per blank_context(state) bucket (see its docstring). An
    action's value genuinely depends on where the blank currently sits
    relative to goal, so a single global counter per action would average
    together situations where that same action helps and where it hurts,
    which drowns the signal in noise. Bucketing by context makes each
    (context, action) pair close to its own stationary bandit arm.

    Formula per context: score = success_rate + exploration_bonus
    where exploration_bonus = sqrt(2 * ln(context_attempts) / action_attempts)

    Note on optimality: this only affects *tie-breaking* among nodes with
    equal f(n) -- it never overrides f(n) itself. Whether the overall
    search is optimal still depends entirely on whether heuristic() is
    admissible, independent of this mechanism.
    """
    import math

    print("\nUsing MEMORY-GUIDED A* with contextual UCB exploration!\n")

    # context_stats[context][action] = {"attempts": int, "successes": int}
    context_stats = {}

    def stats_for(context):
        if context not in context_stats:
            context_stats[context] = {
                action: {"attempts": 1, "successes": 0} for action in ACTIONS
            }
        return context_stats[context]

    def ucb_score(context, action):
        counts = stats_for(context)
        total_attempts = sum(c["attempts"] for c in counts.values())
        attempts = counts[action]["attempts"]
        success_rate = counts[action]["successes"] / attempts
        exploration_bonus = math.sqrt(2 * math.log(total_attempts) / attempts)
        return success_rate + exploration_bonus

    frontier = []
    explored = set()
    frontier_set = set()
    counter = 0
    nodes_traversed = 0
    nodes_expanded = 0

    # Start with initial state
    f_value = initial_state.cost + heuristic(initial_state)
    heapq.heappush(frontier, (f_value, 0, counter, initial_state))
    frontier_set.add(initial_state)

    while frontier:
        _, _, _, node = heapq.heappop(frontier)
        frontier_set.discard(node)

        nodes_traversed += 1

        # Show action statistics every 10 nodes, for the node's own context
        if nodes_traversed % 10 == 0:
            context = blank_context(node)
            counts = stats_for(context)
            total_attempts = sum(c["attempts"] for c in counts.values())
            print(f"\n📊 Action Statistics after {nodes_traversed} nodes (context={context}):")
            for action in ACTIONS:
                score = ucb_score(context, action)
                print(
                    f"  {action}: Success={counts[action]['successes'] / counts[action]['attempts']:.2%}"
                    f" (n={counts[action]['attempts']}) UCB={score:.3f}"
                )

        print(f"\nTraversed Node #{nodes_traversed}:")
        print(node)
        print()

        if goal_test(node):
            print("\nGoal found!")
            print(f"Total Nodes Traversed: {nodes_traversed}")
            print(f"Total Nodes Expanded: {nodes_expanded}")
            print(f"\nContexts observed: {len(context_stats)}")
            return node

        explored.add(node)

        # Expand
        parent_h = heuristic(node)
        parent_context = blank_context(node)
        successors = successor_function(node)
        nodes_expanded += 1

        for action, child in successors:
            if child not in explored and child not in frontier_set:
                # Track, in this context, whether this action improved the heuristic
                child_h = heuristic(child)
                counts = stats_for(parent_context)
                counts[action]["attempts"] += 1
                if child_h < parent_h:
                    counts[action]["successes"] += 1

                f_value = child.cost + child_h
                score = ucb_score(parent_context, action)

                # Use negative UCB score so higher UCB = higher priority
                tie_breaker = -int(score * 100)

                counter += 1
                heapq.heappush(frontier, (f_value, tie_breaker, counter, child))
                frontier_set.add(child)

    print("Search failed. No solution found.")
    print(f"Total Nodes Traversed: {nodes_traversed}")
    print(f"Total Nodes Expanded: {nodes_expanded}")
    return None


# ------------------------------------------------------------
# General Search Algorithm
# ------------------------------------------------------------


def general_search(
    strategy_name: str, initial_state: PuzzleState
) -> Optional[PuzzleState]:
    """
    General search algorithm framework for 8-puzzle.

    Parameters:
      - strategy_name: e.g., "BFS", "DFS", "A*", or your own custom name
      - initial_state: one of EASY/MEDIUM/HARD_INITIAL_STATE

    Students must implement the frontier logic for required strategies.
    You may also add your own strategy (see instructions below).
    """

    # --------------------------------------------------------
    # Initialization (same for all strategies)
    # --------------------------------------------------------
    explored = set()
    nodes_traversed = 0
    nodes_expanded = 0
    frontier = None
    frontier_set = set()  # For A*, track states in frontier for duplicate checking

    # --------------------------------------------------------
    # STRATEGY-SPECIFIC SECTION
    # --------------------------------------------------------
    # Initialize the frontier based on the search strategy.
    # You may use deque, list, heapq, etc., as appropriate.

    if strategy_name == "BFS":
        # TODO: Initialize frontier for BFS (FIFO queue)
        frontier = deque([initial_state])

    elif strategy_name == "DFS":
        # TODO: Initialize frontier for DFS (LIFO stack)
        frontier = deque([initial_state])

    elif strategy_name == "A*":
        # TODO: Initialize frontier for A* (priority queue ordered by f(n) = g(n) + h(n))
        # counter ensures FIFO ordering for ties
        counter = 0
        f_value = initial_state.cost + heuristic(initial_state)
        frontier = [(f_value, counter, initial_state)]
        frontier_set.add(initial_state)

    elif strategy_name == "MEMORY_GUIDED":
        # 🧠 Call the memory-guided A* implementation
        return memory_guided_astar(initial_state)

    # --------------------------------------------------------
    # 💡 EXTENDING WITH NEW STRATEGIES
    # --------------------------------------------------------
    # To add your own strategy:
    #   1. Add a new `elif strategy_name == "NEW_STRATEGIE":` block below.
    #   2. Implement frontier initialization and node selection.
    #   3. Call your strategy in main() like: general_search("NEW_STRATEGIE", initial)

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"To add a new strategy, extend the STRATEGY-SPECIFIC SECTION above."
        )

    # --------------------------------------------------------
    # General Search Loop
    # --------------------------------------------------------
    while frontier:
        # STRATEGY-SPECIFIC: Remove a node from the frontier
        if strategy_name == "BFS":
            # TODO: Pop from the left (FIFO)
            node = frontier.popleft()
        elif strategy_name == "DFS":
            # TODO: Pop from the right (LIFO)
            node = frontier.pop()
        elif strategy_name == "A*":
            # TODO: Pop the node with lowest f(n) = cost + heuristic
            # Pop the node with lowest f(n) = cost + heuristic
            _, _, node = heapq.heappop(frontier)
            frontier_set.discard(node)

        # 💡For custom strategies, add your pop logic here

        if node is None:
            raise RuntimeError(
                "You must assign `node` in the strategy-specific pop step."
            )

        nodes_traversed += 1
        print(f"\nTraversed Node #{nodes_traversed}:\n{node}\n")

        if goal_test(node):
            print("\n Goal found!")
            print(f"Total Nodes Traversed: {nodes_traversed}")
            print(f"Total Nodes Expanded: {nodes_expanded}")
            return node

        explored.add(node)

        # Expand node
        successors = successor_function(node)
        nodes_expanded += 1
        print(
            f"Expanded Node #{nodes_expanded} — Generated {len(successors)} successors.\n"
        )

        for action, child in successors:
            if child not in explored:
                # STRATEGY-SPECIFIC: Add child to frontier
                if strategy_name == "BFS" or strategy_name == "DFS":
                    # TODO: Append child to frontier
                    frontier.append(child)
                elif strategy_name == "A*":
                    # TODO: Add child with priority = child.cost + heuristic(child)
                    # Only add if not already in frontier or if we found a better path
                    if child not in frontier_set:
                        counter += 1
                        f_value = child.cost + heuristic(child)
                        heapq.heappush(frontier, (f_value, counter, child))
                        frontier_set.add(child)

                # 💡For custom strategies, handle insertion here

    print("Search failed. No solution found.")
    print(f"Total Nodes Traversed: {nodes_traversed}")
    print(f"Total Nodes Expanded: {nodes_expanded}")
    return None


# ------------------------------------------------------------
# Helper: Reconstruct Solution Path
# ------------------------------------------------------------


def reconstruct_path(state: PuzzleState) -> List[str]:
    """Reconstruct the sequence of actions from initial to goal."""
    path = []
    while state.parent is not None:
        path.append(state.action)
        state = state.parent
    path.reverse()
    return path


# --- Modified wrappers to collect stats ---
def run_with_stats(strategy_name: str, initial_state: PuzzleState) -> int:
    """
    Runs a search and returns the number of nodes traversed.
    """
    nodes_traversed = 0

    if strategy_name == "A*":
        explored = set()
        frontier = []
        frontier_set = set()
        counter = 0

        f_value = initial_state.cost + heuristic(initial_state)
        heapq.heappush(frontier, (f_value, counter, initial_state))
        frontier_set.add(initial_state)

        while frontier:
            _, _, node = heapq.heappop(frontier)
            frontier_set.discard(node)
            nodes_traversed += 1

            if goal_test(node):
                return nodes_traversed

            explored.add(node)

            for _, child in successor_function(node):
                if child not in explored and child not in frontier_set:
                    counter += 1
                    f_value = child.cost + heuristic(child)
                    heapq.heappush(frontier, (f_value, counter, child))
                    frontier_set.add(child)

    elif strategy_name == "MEMORY_GUIDED":
        # Modified Memory-Guided version that only counts nodes.
        # Mirrors memory_guided_astar()'s contextual-bandit bucketing --
        # see that function's docstring for why context matters.
        import math

        context_stats = {}

        def stats_for(context):
            if context not in context_stats:
                context_stats[context] = {
                    action: {"attempts": 1, "successes": 0} for action in ACTIONS
                }
            return context_stats[context]

        def ucb_score(context, action):
            counts = stats_for(context)
            total_attempts = sum(c["attempts"] for c in counts.values())
            attempts = counts[action]["attempts"]
            success_rate = counts[action]["successes"] / attempts
            exploration_bonus = math.sqrt(2 * math.log(total_attempts) / attempts)
            return success_rate + exploration_bonus

        frontier = []
        explored = set()
        frontier_set = set()
        counter = 0

        f_value = initial_state.cost + heuristic(initial_state)
        heapq.heappush(frontier, (f_value, 0, counter, initial_state))
        frontier_set.add(initial_state)

        while frontier:
            _, _, _, node = heapq.heappop(frontier)
            frontier_set.discard(node)

            nodes_traversed += 1
            if goal_test(node):
                return nodes_traversed

            explored.add(node)

            parent_h = heuristic(node)
            parent_context = blank_context(node)
            successors = successor_function(node)

            for action, child in successors:
                if child not in explored and child not in frontier_set:
                    child_h = heuristic(child)
                    counts = stats_for(parent_context)
                    counts[action]["attempts"] += 1
                    if child_h < parent_h:
                        counts[action]["successes"] += 1

                    score = ucb_score(parent_context, action)
                    tie_breaker = -int(score * 100)

                    counter += 1
                    f_value = child.cost + child_h
                    heapq.heappush(frontier, (f_value, tie_breaker, counter, child))
                    frontier_set.add(child)

    return nodes_traversed


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # Choose your puzzle difficulty
    initial = HARD_INITIAL_STATE  # Try MEDIUM_INITIAL_STATE or HARD_INITIAL_STATE too!

    print("Initial State:")
    print(initial)
    print("\nGoal State:")
    print(GOAL_STATE)

    # Choose your strategy: "BFS", "DFS", "A*", or "MEMORY_GUIDED"!
    result = general_search("A*", initial)

    categories = ["Easy", "Medium", "Hard"]

    if result:
        path = reconstruct_path(result)
        print("\n Solution found!")
        print("Number of steps:", len(path))
        print("Actions:", path)
        print("\nFinal state:")
        print(result)
    else:
        print(" No solution found.")



import matplotlib.pyplot as plt

# Difficulties and initial states
difficulties = ["Easy", "Medium", "Hard"]
initial_states = [EASY_INITIAL_STATE, MEDIUM_INITIAL_STATE, HARD_INITIAL_STATE]

# Collect results
chebyshev_memory_results = []
astar_manhattan_results = []

def run_astar_manhattan(state):
    """Run standard A* with Manhattan heuristic and return nodes traversed."""
    global heuristic
    old_heuristic = heuristic
    heuristic = heuristic_manhattan
    nodes = run_with_stats("A*", state)
    heuristic = old_heuristic
    return nodes

# Run experiments
for state in initial_states:
    nodes_mem = run_with_stats("MEMORY_GUIDED", state)
    chebyshev_memory_results.append(nodes_mem)

    nodes_astar = run_astar_manhattan(state)
    astar_manhattan_results.append(nodes_astar)

# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 6))

x = range(len(difficulties))
width = 0.35

# Color style from your uploaded image
color_astar = "#dd8a47"          # orange
color_memory = "#1f7bf2"         # blue

bars1 = ax.bar(
    [i - width/2 for i in x],
    astar_manhattan_results,
    width,
    label="A* + Manhattan Heuristic",
    color=color_astar
)

bars2 = ax.bar(
    [i + width/2 for i in x],
    chebyshev_memory_results,
    width,
    label="Memory-Guided A* + Chebyshev Heuristic",
    color=color_memory
)

# Axes formatting
ax.set_xticks(list(x))
ax.set_xticklabels(difficulties, fontsize=13)
ax.set_ylabel("Nodes Traversed", fontsize=13)
ax.set_title(
    "8-Puzzle Benchmark: A* vs Memory-Guided A* (by Nodes Traversed)",
    fontsize=16,
    pad=15
)

ax.legend(fontsize=12)

# Add numeric labels above bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + (height * 0.03 + 1),
        f"{height}",
        ha='center',
        va='bottom',
        fontsize=12
    )

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Percentage Improvements
# ------------------------------------------------------------


