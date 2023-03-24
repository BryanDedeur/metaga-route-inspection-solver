import numpy #must be version <1.20.X
import time
import wandb
import os
import random

class DEGA:
    def __init__(self, chrom_len : int, eval_function, log_data_function):
        # Define the print function
        if os.getenv('SILENT_MODE') == '1':
            def print_silent(*args, **kwargs):
                pass
            self.print = print_silent
        else:
            self.print = print

        self.ga_instance = None
        self.chrom_len = chrom_len

        self.log_data_function = log_data_function
        
        def fitness_function(encoding):
            self.num_evaluations += 1
            fitness = eval_function(self, encoding)
            return fitness
        
        self.fitness_function = fitness_function

        self.run_count = 0

        self.run_time_start = 0
        self.run_time_seconds = 0
        self.config = {}
        self.log_data = {}
        self.gen_start_time = -1
        self.run_start_time = -1
        self.num_evaluations = 0    

        self.best_binary = numpy.array([])
        self.best_evaluation = 0
        self.best_generation = 0
        self.best_fitness = 0
        self.best_solution = None
        self.best_heuristics = None
        self.best_time_seconds = 0

    def create(self, seed : int):
        def on_start(ga_instance):
            self.gen_start_time = time.time()
            self.run_start_time = time.time()
            self.print(str(self.run_count) + '. MetaGA run (seed:'+ str(ga_instance.random_seed) +'): [', end = '')

        def on_fitness(ga_instance, population_fitness):

            # self.wandb.log({'min':numpy.min(self.population_objectives),'mean':numpy.mean(self.population_objectives), 'max':numpy.max(self.population_objectives)})
            # population_fitness = numpy.array(population_fitness)
            population_objective = [1/t[1] for t in population_fitness]

            self.log_data['gen min obj'] = numpy.min(population_objective)
            self.log_data['gen mean obj'] = numpy.average(population_objective)
            self.log_data['gen max obj'] = numpy.max(population_objective)

        def on_parents(ga_instance, selected_parents):
            #TODO gather best parents information here
            return

        def on_crossover(ga_instance, offspring_crossover):
            return

        def on_mutation(ga_instance, offspring_mutation):
            return

        def on_generation(ga_instance):

            if (ga_instance.generations_completed % int(ga_instance.num_generations * 0.05) == 0):
                self.print('.', end = '')
            
            self.log_data['gen time(s)'] = time.time() - self.gen_start_time
            self.log_data['total run time(s)'] = time.time() - self.run_start_time

            self.gen_start_time = time.time()
            self.log_data_function(self)


        def on_stop(ga_instance, last_population_fitness):
            self.run_time_seconds = time.time() - self.run_time_start
            self.print('] in ' + str(round(self.run_time_seconds, 3)) + 's')
            self.log_data = {}
            self.log_data['run best evaluation'] = self.best_evaluation
            self.log_data['run best time(s)'] = self.best_time_seconds
            self.log_data['run best generation'] = self.best_generation
            self.log_data['run best obj'] = 1/self.best_fitness
            self.log_data['run best binary'] = ''.join(str(int(x)) for x in numpy.nditer(self.best_binary))
            self.log_data['run best route'] = self.best_solution 
            self.log_data['run best heuristics'] = self.best_heuristics
            self.log_data['run time(s)'] = self.run_time_seconds
            # ga_instance.plot_fitness()

        # create the pygad ga
        self.ga_instance = GeneticAlgorithm(
            num_generations=1000, # GOOD
            num_parents_mating=100, # num parents for mating
            fitness_func=self.fitness_function,
            parent_selection_type='chc',
            crossover_type='ordererd', 
            crossover_probability=0.99, # GOOD
            mutation_type='inversion', # GOOD
            mutation_probability=0.4, # GOOD
            num_genes=self.chrom_len,
            init_range_low=0,
            init_range_high=self.chrom_len,
            gene_type=int,
            random_seed=seed,
            save_best_solutions=False,
            allow_duplicate_genes=False,

            on_start=on_start,
            on_fitness=on_fitness,
            on_parents=on_parents,
            on_crossover=on_crossover,
            on_mutation=on_mutation,
            on_generation=on_generation,
            on_stop=on_stop,
        )

        self.config = {
            'num_generations' : self.ga_instance.num_generations,
            'num_parents_mating' : self.ga_instance.num_parents, 
            'sol_per_pop' : self.ga_instance.pop_size, 
            'parent_selection_type' : self.ga_instance.parent_selection_type,
            'crossover_type' : self.ga_instance.crossover_type, 
            'crossover_probability' : self.ga_instance.crossover_probability,
            'mutation_type' : self.ga_instance.mutation_type,
            'mutation_probability' : self.ga_instance.mutation_probability,
            'num_genes' : self.ga_instance.num_genes,
            'random_seed' : self.ga_instance.random_seed,
        }

    def run(self):
        
        self.population_objectives = numpy.zeros(self.ga_instance.num_parents)

        self.run_count += 1
        self.run_time_start = time.time()
        self.run_time_seconds = 0

        self.best_binary = ''
        self.best_evaluation = 0
        self.best_generation = 0
        self.best_fitness = 0
        self.best_time_seconds = 0

        self.ga_instance.run()

class GeneticAlgorithm:
    def __init__(self, **kwargs):
        self.num_generations = kwargs.get('num_generations', 0)
        self.num_parents = kwargs.get('num_parents_mating', 0)
        self.pop_size = self.num_parents * 2
        self.fitness_fn = kwargs.get('fitness_func', None)
        # self.gene_pool = kwargs.get('init_range_high', None)
        self.gene_length = kwargs.get('num_genes', 0)
        self.gene_pool = list(range(self.gene_length))
        self.num_genes = kwargs.get('num_genes', 0)
        self.parent_selection_type = kwargs.get('parent_selection_type', 'sss')
        self.crossover_type = kwargs.get('crossover_type', 'single_point')
        self.crossover_probability = kwargs.get('crossover_probability', 0.9)
        self.mutation_type = kwargs.get('mutation_type', 'random')
        self.mutation_probability = kwargs.get('mutation_probability', 0.01)
        self.random_seed = kwargs.get('random_seed', None)
        self.generations_completed = 0
        random.seed(self.random_seed)

        self.on_start = kwargs.get('on_start', None)
        self.on_fitness = kwargs.get('on_fitness', None)
        self.on_parents = kwargs.get('on_parents', None)
        self.on_crossover = kwargs.get('on_crossover', None)
        self.on_mutation = kwargs.get('on_mutation', None)
        self.on_generation = kwargs.get('on_generation', None)
        self.on_stop = kwargs.get('on_stop', None)

    def init_population(self):
        return [list(numpy.random.permutation(self.gene_pool)[:self.num_genes]) for _ in range(self.pop_size)]

    def evaluate_population(self, population):
        pop_fitness = [(individual, self.fitness_fn(individual)) for individual in population]
        sorted_pop = sorted(pop_fitness, key=lambda x: x[1], reverse=True)
        self.on_fitness(self, pop_fitness)
        return sorted_pop

    def select_parents(self, population):
        parent_indices = numpy.random.choice(len(population), size=self.num_parents, replace=False)
        parents = [population[i][0] for i in parent_indices]
        population = [individual for i, individual in enumerate(population) if i not in parent_indices]
        self.on_parents(self)
        return parents, population

    def order_crossover(self, parent1, parent2):
        if random.random() > self.crossover_probability:
            # If crossover probability is not met, return a copy of parent1
            return parent1[:]

        start = random.randint(0, self.num_genes-1)
        end = random.randint(start, self.num_genes-1)
        child = parent2[:]
        child[start:end+1] = [gene for gene in parent1[start:end+1] if gene not in child[start:end+1]]
        # Remove duplicates from child sequence
        child_set = set()
        child = [gene if gene not in child_set and not child_set.add(gene) else self.gene_pool[random.randint(0, len(self.gene_pool) - 1)]
                 for gene in child]
        return child

    def cataclysmic_mutation(self, individual):
        if random.random() < self.mutation_probability:
            # Choose a random substring to invert
            start_index = random.randint(0, self.num_genes - 1)
            end_index = random.randint(start_index, self.num_genes - 1)
            # Invert the order of the substring
            individual[start_index:end_index+1] = individual[start_index:end_index+1][::-1]
        # Remove duplicates from individual sequence
        individual_set = set()
        individual = [gene if gene not in individual_set and not individual_set.add(gene) else self.gene_pool[random.randint(0, len(self.gene_pool) - 1)]
                      for gene in individual]
        return individual

    def evolve_population(self, parents):
        num_children = self.pop_size - len(parents)
        children = []
        while len(children) < num_children:
            parent1, parent2 = random.sample(parents, 2)
            child = self.order_crossover(parent1, parent2)
            self.on_crossover(self, child)
            child = self.cataclysmic_mutation(child)
            # self.on_mutation(self)
            children.append(child)
        return children

    def run(self):
        self.on_start(self)
        population = self.init_population()
        for i in range(self.num_generations):
            population = self.evaluate_population(population)
            # print(f"Generation {i+1}: Best fitness - {population[0][1]}")
            parents = [individual for individual, _ in population[:self.num_parents]]
            children = self.evolve_population(parents)
            population = parents + children
            self.on_generation(self)
            self.generations_completed = i + 1
        population = self.evaluate_population(population)
        # print(f"Generation {self.num_generations}: Best individual - {population[0][0]}, Best fitness - {population[0][1]}")
        self.on_stop(self)
