import Reporter
import numpy as np
import random
np.random.seed(seed=30102021)

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		loop_size = len(self.distanceMatrix)
		k = 3
		lambdaa = 1000
		mu = 400
		alpha = 0.2
		mutation_length = 6

		population = self.initialize(loop_size, lambdaa)
		ConvergenceTest = True
		cost_of_population = None
		while(ConvergenceTest):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = None

			# Your code here.
			offspring = []
			for _ in range(mu//2):
				parent1 = self.selection(population, k, cost_of_population)
				parent2 = self.selection(population, k, cost_of_population)
				child1, child2 = self.crossover(parent1, parent2)
				offspring.append(child1)
				offspring.append(child2)
			offspring = np.asarray(list(map(lambda member: self.mutation2(member, alpha, mutation_length), offspring)))
			population = np.asarray(list(map(lambda member: self.mutation2(member, alpha, mutation_length), population)))
			population, cost_of_population = self.elimination(population, offspring, lambdaa)
			invalid_path = np.isinf(cost_of_population)
			population = population[~invalid_path]
			cost_of_population = cost_of_population[~invalid_path]

			meanObjective = cost_of_population.mean()
			bestObjective = cost_of_population.min()
			bestSolution = population[0] # sorted base on cost
			ConvergenceTest = ~np.isclose(bestObjective , meanObjective)
			print(meanObjective, bestObjective)
			print(bestSolution)
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.

		#return 0
		return bestObjective

	def initialize(self, loop_size, lambdaa):
		return np.asarray([self.loop_generator(loop_size) for i in range(lambdaa)])
		
	def loop_generator(self,loop_size):
		# seed?
		loop = np.arange(loop_size)
		loop[1:] = np.random.permutation(loop[1:])
		return loop

	def cost_calculator(self, path):
		cost = 0
		for i, j in zip(path[:-1], path[1:]):
			cost += self.distanceMatrix[i,j]
		cost += self.distanceMatrix[path[-1], path[0]]
		return cost

	def selection(self, population, k, cost_of_population = None):
		if cost_of_population is None:
			selected_index = np.random.choice(len(population), k)
			selected_path = population[selected_index]	
			lowcost_index = np.argmin(list(map(self.cost_calculator, selected_path)))
		else:
			selected_index = np.random.choice(len(population), k)
			selected_path = population[selected_index]	
			lowcost_index = np.argmin(cost_of_population[selected_index])
		return(selected_path[lowcost_index])

	def mutation1(self, path, alpha, length):
		if np.random.rand() < alpha:
			starting_index = np.random.choice(np.arange(1, len(path)-length))
			ending_index = starting_index + length
			path[starting_index: ending_index] = path[starting_index: ending_index][::-1]
		return path

	def mutation2(self, path, alpha, length):
		if np.random.rand() < alpha:
			starting_index1 = np.random.choice(np.arange(1, len(path)-length))
			ending_index1 = starting_index1 + length
			tmp_list = np.empty(0, dtype = int)
			tmp_list = np.append(tmp_list, np.arange(1, starting_index1-length))
			tmp_list = np.append(tmp_list, np.arange(ending_index1, len(path)-length))
			starting_index2 = np.random.choice(tmp_list)
			ending_index2 = starting_index2 + length
			tmp = path[starting_index1: ending_index1].copy()
			path[starting_index1: ending_index1] = path[starting_index2: ending_index2].copy()
			path[starting_index2: ending_index2] = tmp
		return path	

	# lambda + mu elimination
	def elimination(self, population, offspring, lambdaa):
		new_population = np.vstack((population, offspring))
		costs = np.array(list(map(self.cost_calculator, new_population)))
		sort_index = np.argsort(costs)
		new_population = new_population[sort_index]
		cost_of_population = costs[sort_index]
		return new_population[:lambdaa], cost_of_population[:lambdaa]

	def crossover(self, path1, path2):
		length = np.random.randint(2, high=28)
		def operator(P1, P2):
			offspring = -1*np.ones(len(P1), dtype = int)
			offspring[0] = 0
			offspring[starting_index: ending_index] = P1[starting_index: ending_index]
			for i in range(starting_index, ending_index):
				value_assign = P2[i]
				if value_assign in offspring:
					continue
				value_position = P1[i]
				while value_position in P2[starting_index:ending_index]:
					value_position = P1[P2==value_position]
				offspring[P2==value_position] = value_assign
			offspring[offspring==-1] = P2[offspring==-1]
			return offspring

		starting_index = np.random.choice(np.arange(1, len(path1)-length))	
		ending_index = starting_index + length
		
		return operator(path1, path2), operator(path2, path1)

flnm = 'tour29.csv'
c = r0123456()
c.optimize(flnm)
#number_of_test = 10
#bo_list = np.empty(number_of_test)
#for it in range(number_of_test):
#	bo_list[it] = c.optimize(flnm)
#print(bo_list.mean())
