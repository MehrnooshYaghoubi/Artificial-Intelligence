"""
AI Project (2): Traveling Salesman Problem (TSP)
--------------------------------------------
In this project, you will implement SEVEN different algorithms to solve the TSP.

Algorithms to implement:
1. Backtracking
2. Forward Checking (CSP)
3. MVC (Minimum Remaining Values with Constraint Checking)
4. MRV (Most Restricted Variable heuristic)
5. Hill Climbing
6. Simulated Annealing
7. Genetic Algorithm

Each algorithm has a placeholder (TODO) you must complete.
You should test and compare their performance on the same dataset.
"""

import random
import math
import json


class TSPSolver:
    def __init__(self, cities, distances):
        """
        cities: list of city names
        distances: dictionary where distances[(A, B)] = distance
        """
        self.cities = cities
        self.distances = distances

    # -------------------------------------------------------------------------
    # 1 Backtracking
    # -------------------------------------------------------------------------

    def solve_backtracking(self):
        """
        Explore all possible routes using recursive backtracking with pruning.
        """
        if not self.cities:
            return ([], 0)

        best_route = [None]
        best_distance = [
            float("inf")
        ]  # best_distance is a list becasue it can be updated inside backtrack without Python complaining about variable scope

        def backtrack(current_route, unvisited):
            """
            Recursive backtracking helper.

            Args:
                current_route: list of cities visited so far
                unvisited: set of cities not yet visited
            """

            # Base case: all cities visited
            if not unvisited:
                # Calculate total distance S complete route
                total_dist = self.total_distance(current_route)

                # Update best route if this route is better
                if total_dist < best_distance[0]:
                    best_distance[0] = total_dist
                    best_route[0] = current_route[:]
                return

            # Try visiting each unvisited city
            for next_city in unvisited:
                # pruning: estimate lower bound - skip if current path already exceeds best
                current_dist = self.total_distance(current_route + [next_city])
                if current_dist >= best_distance[0]:
                    continue

                # Recursive call: visit next_city
                new_unvisited = unvisited - {next_city}
                backtrack(current_route + [next_city], new_unvisited)

        # Start backtracking from first city
        current_route = [self.cities[0]]
        unvisited = set(self.cities[1:])
        backtrack(current_route, unvisited)

        return (best_route[0], best_distance[0])

    # -------------------------------------------------------------------------
    # 2️ Forward Checking (CSP-based)
    # -------------------------------------------------------------------------

    def solve_forward_checking(self):
        """
        CSP approach: use forward checking to ensure each city is only visited once.
        """
        if not self.cities:
            return ([], 0)

        best_route = [None]
        best_distance = [float("inf")]

        def forward_check(current_route, domains):
            """
            Forward checking: remove visited cities from future domains.

            Args:
                current_route: list of cities visited so far
                domains: dict mapping position to set of available cities
            """
            visited = set(current_route)
            position = len(current_route)

            # Remove visited cities from all future positions
            for pos in range(position, len(self.cities)):
                domains[pos] -= visited

            # Check for domain (empty domain = no solution possible)
            for pos in range(position, len(self.cities)):
                if not domains[pos]:
                    return False
            return True

        def backtrack(current_route, domains):
            """
            Recursive backtracking with forward checking.

            Args:
                current_route: list of cities visited so far
                domains: dict mapping position to set of available cities
            """
            # Base case: all cities visited
            if len(current_route) == len(self.cities):
                total_dist = self.total_distance(current_route)
                if total_dist < best_distance[0]:
                    best_distance[0] = total_dist
                    best_route[0] = current_route[:]
                return

            position = len(current_route)

            # Try each city available at this position
            for city in sorted(domains[position]):
                new_route = current_route + [city]

                # Copy domains for this branch
                new_domains = {i: domains[i].copy() for i in domains}

                # Apply forward checking
                if forward_check(new_route, new_domains):
                    backtrack(new_route, new_domains)

        # Initialize domains: each position can have any city initially
        domains = {i: set(self.cities) for i in range(len(self.cities))}
        backtrack([], domains)

        return (
            (best_route[0], best_distance[0]) if best_route[0] else (None, float("inf"))
        )

    # -------------------------------------------------------------------------
    # 3️ MVC (Minimum Remaining Constraints heuristic)
    # -------------------------------------------------------------------------

    def solve_mvc(self):
        """
        Use Minimum Remaining Values heuristic to guide search.
        """
        
        if not self.cities:
            return ([], 0)
        
        best_route = [None]
        best_distance = [float('inf')]
        
        def backtrack(current_route, domains):
            # Base case: all cities visited
            if len(current_route) == len(self.cities):
                total_dist = self.total_distance(current_route)
                if total_dist < best_distance[0]:
                    best_distance[0] = total_dist
                    best_route[0] = current_route[:]
                return
            
            # Select next position using MRV heuristic (minimum remaining values)
            unassigned = {pos: domains[pos] for pos in domains if pos >= len(current_route)}
            if not unassigned:
                return
            position = min(unassigned, key=lambda k: len(unassigned[k]))
            
            # Try each city at this position
            for city in sorted(domains[position]):
                new_route = current_route + [city]
                
                # Copy domains and remove assigned city from future positions
                new_domains = {i: domains[i].copy() for i in domains}
                new_domains[position] = {city}
                
                # Apply forward checking
                visited = set(new_route)
                for pos in range(len(new_route), len(self.cities)):
                    new_domains[pos] -= visited
                
                # Check for domain wipeout
                valid = True
                for pos in range(len(new_route), len(self.cities)):
                    if not new_domains[pos]:
                        valid = False
                        break
                
                if valid:
                    backtrack(new_route, new_domains)
        
        # Initialize domains
        domains = {i: set(self.cities) for i in range(len(self.cities))}
        backtrack([], domains)
        
        return (best_route[0], best_distance[0]) if best_route[0] else (None, float('inf'))

    # -------------------------------------------------------------------------
    # 4️ MRV (Most Restricted Variable)
    # -------------------------------------------------------------------------

    def solve_mrv(self):
        """
        Use Most Restricted Variable heuristic to decide which city to assign next.
        """
        if not self.cities:
            return ([], 0)
        
        best_route = [None]
        best_distance = [float('inf')]
        
        def backtrack(current_route, domains):
            # Base case: all cities visited
            if len(current_route) == len(self.cities):
                total_dist = self.total_distance(current_route)
                if total_dist < best_distance[0]:
                    best_distance[0] = total_dist
                    best_route[0] = current_route[:]
                return
            
            # Select position with most restricted domain (MRV heuristic)
            unassigned = {pos: domains[pos] for pos in domains if pos >= len(current_route)}
            if not unassigned:
                return
            position = min(unassigned, key=lambda k: len(unassigned[k]))
            
            # Try each city at this position
            for city in sorted(domains[position]):
                new_route = current_route + [city]
                
                # Copy domains
                new_domains = {i: domains[i].copy() for i in domains}
                new_domains[position] = {city}
                
                # Forward checking: remove visited cities from future positions
                visited = set(new_route)
                for pos in range(len(new_route), len(self.cities)):
                    new_domains[pos] -= visited
                
                # Check for domain wipeout
                valid = True
                for pos in range(len(new_route), len(self.cities)):
                    if not new_domains[pos]:
                        valid = False
                        break
                
                if valid:
                    backtrack(new_route, new_domains)
        
        # Initialize domains
        domains = {i: set(self.cities) for i in range(len(self.cities))}
        backtrack([], domains)
        
        return (best_route[0], best_distance[0]) if best_route[0] else (None, float('inf'))

    # -------------------------------------------------------------------------
    # 5️ Hill Climbing
    # -------------------------------------------------------------------------

    def solve_hill_climbing(self, max_iterations=1000):
        """
        Start with a random solution and iteratively move to a better neighbor.
        """
        current_solution = self.cities[:]
        random.shuffle(current_solution)
        current_distance = self.total_distance(current_solution)

        for _ in range(max_iterations):
            neighbors = [
                current_solution[:i]
                + [current_solution[j]]
                + current_solution[i + 1 : j]
                + [current_solution[i]]
                + current_solution[j + 1 :]
                for i in range(len(current_solution))
                for j in range(i + 1, len(current_solution))
            ]

            best_neighbor, best_distance = min(
                ((neighbor, self.total_distance(neighbor)) for neighbor in neighbors),
                key=lambda x: x[1],
                default=(None, float("inf")),
            )

            if best_distance < current_distance:
                current_solution, current_distance = best_neighbor, best_distance
            else:
                break

        return current_solution, current_distance

    # -------------------------------------------------------------------------
    # 6️ Simulated Annealing
    # -------------------------------------------------------------------------

    def solve_simulated_annealing(
        self, initial_temp=1000, cooling_rate=0.995, stop_temp=1
    ):
        """
        Simulated Annealing: probabilistically accept worse solutions to escape local minima.
        """

        def accept_probability(current_distance, neighbor_distance, temperature):
            """Calculate the probability of accepting a worse solution."""
            if neighbor_distance < current_distance:
                return 1.0
            return math.exp((current_distance - neighbor_distance) / temperature)

        # Start with a random route
        current_solution = self.cities[:]
        random.shuffle(current_solution)
        current_distance = self.total_distance(current_solution)

        best_solution = current_solution[:]
        best_distance = current_distance

        temperature = initial_temp

        while temperature > stop_temp:
            # Generate a neighbor by swapping two cities
            i, j = random.sample(range(len(current_solution)), 2)
            neighbor_solution = current_solution[:]
            neighbor_solution[i], neighbor_solution[j] = (
                neighbor_solution[j],
                neighbor_solution[i],
            )
            neighbor_distance = self.total_distance(neighbor_solution)

            # Accept or reject the neighbor solution
            if random.random() < accept_probability(
                current_distance, neighbor_distance, temperature
            ):
                current_solution = neighbor_solution
                current_distance = neighbor_distance

                # Update the best solution found so far
                if current_distance < best_distance:
                    best_solution = current_solution[:]
                    best_distance = current_distance

            # Gradually decrease the temperature
            temperature *= cooling_rate

        return best_solution, best_distance

    # -------------------------------------------------------------------------
    # 7️ Genetic Algorithm
    # -------------------------------------------------------------------------

    def solve_genetic(self, population_size=50, generations=200, mutation_rate=0.1):
        """
        Genetic Algorithm: evolve a population of routes to find a near-optimal solution.
        """

        def initialize_population():
            """Generate a random population of routes."""
            return [
                random.sample(self.cities, len(self.cities))
                for _ in range(population_size)
            ]

        def fitness(route):
            """Calculate fitness as the inverse of the total distance."""
            return 1 / self.total_distance(route)

        def select_parents(population):
            """Select two parents using a fitness-proportionate selection (roulette wheel)."""
            fitness_values = [fitness(route) for route in population]
            total_fitness = sum(fitness_values)
            probabilities = [f / total_fitness for f in fitness_values]
            return random.choices(population, probabilities, k=2)

        def crossover(parent1, parent2):
            """Perform ordered crossover to produce an offspring."""
            start, end = sorted(random.sample(range(len(parent1)), 2))
            child = [None] * len(parent1)
            child[start:end] = parent1[start:end]

            pointer = 0
            for city in parent2:
                if city not in child:
                    while child[pointer] is not None:
                        pointer += 1
                    child[pointer] = city

            return child

        def mutate(route):
            """Perform mutation by swapping two cities with a given probability."""
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]

        # Initialize population
        population = initialize_population()

        best_route = None
        best_distance = float("inf")

        for _ in range(generations):
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = select_parents(population)
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)

                mutate(child1)
                mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

            # Update the best solution found so far
            for route in population:
                distance = self.total_distance(route)
                if distance < best_distance:
                    best_route = route
                    best_distance = distance

        return best_route, best_distance

    # -------------------------------------------------------------------------
    # 7️ Total Distance Calculation
    # -------------------------------------------------------------------------

    def total_distance(self, path):
        """Compute total route distance (including return to start)."""
        total = 0
        for i in range(len(path) - 1):
            total += self.distances[(path[i], path[i + 1])]
        total += self.distances[(path[-1], path[0])]
        return total


# -------------------------------------------------------------------------
# Input loader and main
# -------------------------------------------------------------------------
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cities"], {(a, b): d for (a, b, d) in data["distances"]}


if __name__ == "__main__":
    cities, distances = load_data("sample_input.json")
    solver = TSPSolver(cities, distances)

    print("Choose Algorithm:")
    print("1. Backtracking")
    print("2. Forward Checking")
    print("3. MVC")
    print("4. MRV")
    print("5. Hill Climbing")
    print("6. Simulated Annealing")
    print("7. Genetic Algorithm")

    choice = input(" Your Algorithm Number: ")

    if choice == "1":
        result = solver.solve_backtracking()
    elif choice == "2":
        result = solver.solve_forward_checking()
    elif choice == "3":
        result = solver.solve_mvc()
    elif choice == "4":
        result = solver.solve_mrv()
    elif choice == "5":
        result = solver.solve_hill_climbing()
    elif choice == "6":
        result = solver.solve_simulated_annealing()
    elif choice == "7":
        result = solver.solve_genetic()
    else:
        print("Not Valid")
        exit()

    print("\nResult:", result)
