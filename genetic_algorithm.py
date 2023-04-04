import numpy as np
import matplotlib.pyplot as plt
from random import randint, uniform
import math

# define the sine function
def sine(x):
    return x*np.sin(10*np.pi*x)+2

def b_to_d(num):
    sum = 0 
    n = 6
    for i in num:
        
        sum += int(i) * math.pow(2,n)
        n = n-1
    sum = ((sum/math.pow(2,6)))-1
    if uniform(0, 1) < 0.5:
        return -sum
    return sum

# define the fitness function
def fitness_function(population):
    # calculate the fitness of each individual
    fitness = []
    for individual in population:
        x = np.linspace(-1, 1, len(individual))
        y = sine(x)
        i = b_to_d(individual)
        y1 = sine(i)
        error = np.mean(np.abs(y - y1))
        fitness.append(1 / (1 + error))
    return fitness

# define the crossover function
def crossover(parent1, parent2):
    # randomly select a crossover point
    crossover_point = randint(0, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# define the mutation function
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if uniform(0, 1) < mutation_rate:
            individual[i] = uniform(-1, 1)
    return individual

# define the genetic algorithm function
def genetic_algorithm(population_size, num_generations, mutation_rate):
    # generate the initial population
    population = []
    for i in range(population_size):
        individual = np.random.uniform(-1,1,100)
        #print(individual)
        population.append(individual)

    # iterate through the generations
    for generation in range(num_generations):
        # evaluate the fitness of each individual
        fitness = fitness_function(population)

        
        for i in range (0,len(population)-1,2):
            
            parent1 = population[i]
            parent2 = population[i+1]
            # create the offspring using crossover
            child1, child2 = crossover(parent1, parent2)
    
            # apply mutation to the offspring
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            fitness = fitness_function(population)
            population.append(child1)
            population.append(child2)
            
        fit = np.argsort(fitness)[::-1]
        
        for i in range (population_size):
            population[i] = population[fit[i]]
        
        population = population[:population_size]
      

    # return the best individual in the final population
    fitness = fitness_function(population)
    best_individual = population[np.argmax(fitness)]
    return best_individual

# run the genetic algorithm and plot the results
best_individual = genetic_algorithm(population_size=10, num_generations=1000, mutation_rate=0.1)
best_individual.sort()

x = np.linspace(-1, 1,5000)
y = sine(x)
plt.plot(x, y, label="actual")
plt.plot(best_individual, sine(best_individual), label="approximation")
plt.legend()
plt.show()