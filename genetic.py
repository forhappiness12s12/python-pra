import numpy as np

# Parameters
population_size = 100
mutation_rate = 0.01
generations = 100
chromosome_length = 10

# Objective function
def objective_function(x):
    return -np.sum(x**2)

# Generate initial population
def generate_population(size, length):
    return np.random.randint(0, 2, (size, length))

# Fitness function
def fitness(population):
    return np.array([objective_function(individual) for individual in population])

# Selection
def select(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[-population_size//2:]
    return population[selected_indices]

# Crossover
def crossover(parent1, parent2):
    point = np.random.randint(1, chromosome_length - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

# Mutation
def mutate(individual):
    for i in range(chromosome_length):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# Main genetic algorithm
def genetic_algorithm():
    population = generate_population(population_size, chromosome_length)
    for generation in range(generations):
        fitness_scores = fitness(population)
        selected_population = select(population, fitness_scores)
        next_population = []

        while len(next_population) < population_size:
            parents = selected_population[np.random.choice(selected_population.shape[0], 2, replace=False)]
            child1, child2 = crossover(parents[0], parents[1])
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))
        
        population = np.array(next_population)
        best_fitness = np.max(fitness_scores)
        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    best_individual = population[np.argmax(fitness(population))]
    return best_individual

# Run the genetic algorithm
best_solution = genetic_algorithm()
print(f'Best Solution: {best_solution}')
print(f'Best Fitness: {objective_function(best_solution)}')
