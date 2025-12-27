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
    # Simulated Annealing
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
    # All Functions
    # -------------------------------------------------------------------------

    # 1. Geometric Cooling
    def geometric_cooling(T_k, alpha):
        """
        T_{k+1} = alpha * T_k
        """
        return alpha * T_k

    # 2. Linear Cooling
    def linear_cooling(T_k, beta):
        """
        T_{k+1} = T_k - beta
        """
        return T_k - beta

    # 3. Logarithmic Cooling
    def logarithmic_cooling(T_0, k, c=1):
        """
        T_k = T_0 / log(k + c)
        """
        return T_0 / math.log(k + c)

    # 4. Exponential Cooling
    def exponential_cooling(T_0, k, lambd):
        """
        T_k = T_0 * exp(-lambda * k)
        """
        return T_0 * math.exp(-lambd * k)

    # 5. Fast Cooling
    def fast_cooling(T_0, k):
        """
        T_k = T_0 / k
        """
        return T_0 / k

    # 6. Adaptive Cooling (Acceptance-Rate Based)
    def adaptive_cooling(T_k, r_k, r_star, gamma):
        """
        T_{k+1} =
        T_k * (1 + gamma), if r_k > r_star
        T_k * (1 - gamma), if r_k < r_star
        """
        if r_k > r_star:
            return T_k * (1 + gamma)
        elif r_k < r_star:
            return T_k * (1 - gamma)
        return T_k

    # 7. Cost-Difference Feedback
    def cost_difference_feedback(T_k, delta_E_k, gamma):
        """
        T_{k+1} = T_k * (1 + gamma * delta_E_k)
        """
        return T_k * (1 + gamma * delta_E_k)

    # 8. Reheating
    def reheating(T_0, eta):
        """
        T_{k+1} = eta * T_0
        """
        return eta * T_0

    # 9. Hybrid Schedule
    def hybrid_schedule(T_k, alpha, r_k, f):
        """
        T_{k+1} = alpha * T_k * f(r_k)
        """
        return alpha * T_k * f(r_k)

    # 10. Stochastic Temperature
    def stochastic_temperature(T_0, k):
        """
        T_k = T_0 * exp(-xi_k * k), where xi_k ~ U(0, 1)
        """
        xi_k = random.uniform(0, 1)
        return T_0 * math.exp(-xi_k * k)




    # -------------------------------------------------------------------------
    # Total Distance Calculation
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

    print("\nResult:", solver.solve_simulated_annealing())
