import base58
import numpy as np
from ecdsa import SigningKey, SECP256k1

# Parameters
population_size = 100
mutation_rate = 0.01
generations = 100
private_key_length = 64  # Length of the private key in hexadecimal

# Target public key (Base58 encoded)
base58_public_key = 'TFKSWmnRJkzCfWUr56DQ1zqGCxZLW9dKSv'  # Replace with your actual target public key
target_public_key = base58.b58decode(base58_public_key)

# Generate initial population
def generate_population(size, length):
    return np.array([np.random.bytes(length//2).hex() for _ in range(size)])

# Fitness function
def fitness_function(public_key):
    # Ensure both public_key and target_public_key are the same length
    min_length = min(len(public_key), len(target_public_key))
    public_key = public_key[:min_length]
    target_public_key = target_public_key[:min_length]
    
    # Calculate fitness based on Hamming distance
    return np.sum(np.array(list(public_key)) != np.array(list(target_public_key)))

# Generate public key from private key
def private_key_to_public_key(private_key):
    private_key_bytes = bytes.fromhex(private_key)
    sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
    vk = sk.get_verifying_key()
    public_key_bytes = vk.to_string('compressed')  # Use compressed format
    return public_key_bytes

# Crossover
def crossover(parent1, parent2):
    point = np.random.randint(1, private_key_length - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation
def mutate(individual):
    mutated = list(individual)
    for i in range(private_key_length):
        if np.random.rand() < mutation_rate:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)

# Main genetic algorithm
def genetic_algorithm():
    population = generate_population(population_size, private_key_length)
    for generation in range(generations):
        fitness_scores = []
        for individual in population:
            public_key = private_key_to_public_key(individual)
            fitness_scores.append(fitness_function(public_key))
        fitness_scores = np.array(fitness_scores)
        selected_indices = np.argsort(fitness_scores)[:population_size//2]
        selected_population = population[selected_indices]

        next_population = []
        while len(next_population) < population_size:
            parents = selected_population[np.random.choice(selected_population.shape[0], 2, replace=False)]
            child1, child2 = crossover(parents[0], parents[1])
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        population = np.array(next_population)
        best_fitness = np.min(fitness_scores)
        print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

    best_individual = population[np.argmin(fitness_scores)]
    return best_individual

# Run the genetic algorithm
best_solution = genetic_algorithm()
print(f'Best Private Key: {best_solution}')
best_public_key = private_key_to_public_key(best_solution)
print(f'Best Public Key: {best_public_key.hex()}')
print(f'Fitness: {fitness_function(best_public_key)}')
