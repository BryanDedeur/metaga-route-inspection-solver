import numpy #must be version <1.20.X
import pygad
import time
import wandb
import os

class MetaGA:
    def __init__(self, gene_len : int, chrom_len : int, eval_function, log_data_function):
        # Define the print function
        if os.getenv('SILENT_MODE') == '1':
            def print_silent(*args, **kwargs):
                pass
            self.print = print_silent
        else:
            self.print = print

        self.ga_instance = None
        self.gene_len = gene_len
        self.chrom_len = chrom_len

        self.log_data_function = log_data_function
        
        def fitness_function(encoding, individual_id):
            self.num_evaluations += 1
            fitness = eval_function(self, encoding, individual_id)
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
            population_objective = 1/population_fitness

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
        self.ga_instance = pygad.GA(
            num_generations=150,

            num_parents_mating=100, # num parents for mating
            fitness_func=self.fitness_function,
            sol_per_pop=100, # population size including parents and children
            parent_selection_type='rank',
            # K_tournament=50, # no effect unless using k_tournament
            # keep_parents=100, # double check this one (no effect if keep_elitism = 0)
            keep_elitism=50,
            crossover_type='two_points',
            crossover_probability=0.99,
            mutation_type='inversion',
            mutation_probability=0.1,
            num_genes=self.chrom_len,
            init_range_low=0,
            init_range_high=2,
            gene_type=int,
            random_seed=seed,
            save_best_solutions=False,

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
            'num_parents_mating' : self.ga_instance.num_parents_mating, 
            'sol_per_pop' : self.ga_instance.sol_per_pop, 
            'parent_selection_type' : self.ga_instance.parent_selection_type,
            'K_tournament' : self.ga_instance.K_tournament,
            'keep_parents' : self.ga_instance.keep_parents,
            'keep_elitism' : self.ga_instance.keep_elitism,
            'crossover_type' : self.ga_instance.crossover_type, 
            'crossover_probability' : self.ga_instance.crossover_probability,
            'mutation_type' : self.ga_instance.mutation_type,
            'mutation_probability' : self.ga_instance.mutation_probability,
            'num_genes' : self.ga_instance.num_genes,
            'random_seed' : self.ga_instance.random_seed,
        }

    def run(self):
        
        self.population_objectives = numpy.zeros(self.ga_instance.sol_per_pop)

        self.run_count += 1
        self.run_time_start = time.time()
        self.run_time_seconds = 0

        self.best_binary = ''
        self.best_evaluation = 0
        self.best_generation = 0
        self.best_fitness = 0
        self.best_time_seconds = 0

        self.ga_instance.run()

