from random import randint
import tsp_reader as reader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#should we initiate pheromone value?

EVAP_RATE = 0.5 #0.2 or 0.8 seems to be best (see graph)
NBR_ANTS = 500
#alpha is relative importance of pheromone
ALPHA = 1
#beta is relative importance heuristic
BETA = 5
Q = 500
#Q is the constant used to update the pheromones (Q/L)
ITERATIONS = 100

class graph:
    def __init__(self, distances):
        self.distances = distances
        self.pheromones = []


    def set_constants( self, evap_rate, nbr_ants, q, alpha, beta):
        self.nbr_ants = nbr_ants
        self.evap_rate = evap_rate
        self.q = q
        self.alpha = alpha
        self.beta = beta

    def iterations(self, nbr_iter,evap_rate, nbr_ants, q, alpha, beta):
        self.set_constants(evap_rate, nbr_ants, q, alpha, beta)
        self.pheromones = []
        for _ in range(len(self.distances)):
            #inititing to 1 or I divide by 0
            self.pheromones.append([1]*len(self.distances))
        for __ in range(nbr_iter):
            self.make_new_ants()    
            #-1 because first city is already visited
            for _ in range(len(self.distances)-1):
                self.move_ants()
            self.update_pheromones()
    
    
    def get_ants(self):
        return self.ants
    def get_pheromones(self):
        return self.pheromones
    def get_distances(self):
        return self.distances

    def make_new_ants(self):
        self.ants = []
        for _ in range(self.nbr_ants):
            self.ants.append(ant(self.get_distances(),self.alpha,self.beta))

    def add_ant_pheromone(self,n,m, pheromone):
        self.pheromones[n][m] = (1-self.evap_rate)*pheromone+self.pheromones[n][m]
        
    def update_pheromones(self):
        for ant in self.ants:
            visit = ant.visited_cities
            #-1 because i+1
            for i in range(len(visit)-1):
                n = visit[i]
                m = visit[i+1]
                self.add_ant_pheromone(n,m,self.q/int(self.distances[n][m]))

    def move_ants(self):
        for ant in self.get_ants():
            choice = ant.choose_city(self.get_pheromones())
            ant.move_to_city(choice)


    def get_ants_paths(self):
        paths = []
        for ant in self.ants:
            paths.append(ant.get_visited_cities())
        return paths
        
    def get_best_ant(self):
        shorthest_path = self.ants[0].get_final_distance_travelled()
        best_ant = self.ants[0]
        for ant in self.ants:
            ant_dist_travelled = ant.get_final_distance_travelled()
            if(shorthest_path > ant_dist_travelled):
                shorthest_path = ant_dist_travelled
                best_ant = ant
        return best_ant

    def get_path_length(self,path):
        distance_traveled = 0
        for i in range (len(path)-1):
            distance_traveled +=  int(self.distances[path[i]][path[i+1]])
        return distance_traveled
                 
    






    
        
  
class ant:
    def __init__(self, distances,alpha,beta):
        self.nbr_of_cities = len(distances)
        self.distances = distances
        self.visited_cities = [0]
        self.cities_to_visit = []
        self.alfa = alpha
        self.beta = beta
        for i in range (1,self.nbr_of_cities):
            self.cities_to_visit.append(i)
        #starting_city = randint(0,len(distances)-1)
        #starting_city = 0
        #self.cities_to_visit.remove(starting_city)
        #self.visited_cities.append(starting_city)
    
    def get_current_city(self):
        return self.visited_cities[-1]
    def get_distance(self,i,j):
        return int(self.distances[i][j])
    def get_visited_cities(self):
        return self.visited_cities
    def get_final_distance_travelled(self):
        visited_cities = self.get_visited_cities()
        distance_traveled = 0
        for i in range (self.nbr_of_cities -1):
            distance_traveled +=  self.get_distance(visited_cities[i],visited_cities[i+1])
        return distance_traveled


    def move_to_city(self, city):
        self.visited_cities.append(city)
        self.cities_to_visit.remove(city)

    def choose_city(self,pheromones):
        #rdm = randint(0,100)
        #last_rdms =0
        current_city = self.get_current_city()
        sum_teta_nue = 0
        probs = []  
        for city in self.cities_to_visit:
            sum_teta_nue += self.compute_teta_nue(current_city,city,
            pheromones[current_city][city])
        for city in self.cities_to_visit:
            prob = self.compute_teta_nue(current_city,city,
            (pheromones[current_city][city]))/sum_teta_nue
            probs.append(prob)

        #return self.cities_to_visit[-1]   
        return np.random.choice(self.cities_to_visit, 1, p=probs)[0]
               

            
    def compute_teta_nue(self,i,j,pheromone):
        teta = (pheromone)**ALPHA
        nue = (1/int(self.get_distance(i,j)))**BETA
        return teta*nue

distances = reader.tsp_reader("bays29.tsp")

graph = graph(distances)

graph.iterations(ITERATIONS,EVAP_RATE, NBR_ANTS, Q, ALPHA, BETA)
graph.get_ants_paths()
best_ant = graph.get_best_ant()
print(best_ant.get_visited_cities())
print(best_ant.get_final_distance_travelled())
opt_path = reader.opt_reader("bays29.opt.tour")
print(opt_path)
print(graph.get_path_length(opt_path))

file =  open("res_ant.txt", "w")
opt_path_length = graph.get_path_length(opt_path)
best_path_found = best_ant.get_visited_cities()
best_distance_travelled = best_ant.get_final_distance_travelled()

file.write("opt path="+ str(opt_path) + "\n" + "opt length= "+ str(opt_path_length)
            +"\n"+ "best path found= " + str(best_path_found) +"\n"+
            "best distance travelled= " + str(best_distance_travelled))






##### making graph of iterations ############
# possible_iterations = [1,5,10,25,50,100,200,500]
# y_iterations = []
# graph = graph(distances)
# opt_path = reader.opt_reader("bays29.opt.tour")
# opt_length = graph.get_path_length(opt_path)

# for i in [1,5,10,25,50,100,200,500]:
#     graph.iterations(i, EVAP_RATE, NBR_ANTS, Q, ALPHA, BETA)
#     graph.get_ants_paths()
#     best_ant = graph.get_best_ant()
#     #print(best_ant.get_visited_cities())
#     print(best_ant.get_final_distance_travelled())
#     y_iterations.append(best_ant.get_final_distance_travelled())

# plt.plot(possible_iterations,y_iterations, label="ants best length")
# plt.plot(possible_iterations,[opt_length]*len(possible_iterations),label="theoritical optimal length")

# plt.xlabel("number of iterations")
# plt.ylabel("length of path")
# plt.title("bays29 evolution")
# plt.legend()
# plt.savefig('bays29_iterations.png')


##### making graph of evap_rates ############

# possible_evap_rate = np.arange(0,1,0.1)
# y_evap_rate = []
# graph = graph(distances)
# opt_path = reader.opt_reader("bays29.opt.tour")
# opt_length = graph.get_path_length(opt_path)

# for evap_rate in possible_evap_rate:
#     graph.iterations(5, evap_rate, NBR_ANTS, Q, ALPHA, BETA)
#     graph.get_ants_paths()
#     best_ant = graph.get_best_ant()
#     #print(best_ant.get_visited_cities())
#     print(best_ant.get_final_distance_travelled())
#     y_evap_rate .append(best_ant.get_final_distance_travelled())

# plt.plot(possible_evap_rate ,y_evap_rate , label="ants best length")
# plt.plot(possible_evap_rate ,[opt_length]*len(possible_evap_rate),label="theoritical optimal length")

# plt.xlabel("evaporation rate")
# plt.ylabel("length of path")
# plt.title("bays29 evaporation rate evolution")
# plt.legend()
# plt.savefig('bays29_evap_rate.png')

######## making graph of ant numbers ############

# possible_ant_numbers = np.arange(100,1000,100)
# y_ant_nbr = []
# graph = graph(distances)
# opt_path = reader.opt_reader("bays29.opt.tour")
# opt_length = graph.get_path_length(opt_path)

# for ant_nbrs in possible_ant_numbers:
#     graph.iterations(200, EVAP_RATE, ant_nbrs, Q, ALPHA, BETA)
#     graph.get_ants_paths()
#     best_ant = graph.get_best_ant()
#     #print(best_ant.get_visited_cities())
#     print(best_ant.get_final_distance_travelled())
#     y_ant_nbr.append(best_ant.get_final_distance_travelled())

# plt.plot(possible_ant_numbers ,y_ant_nbr , label="ants best length")
# plt.plot(possible_ant_numbers ,[opt_length]*len(possible_ant_numbers),label="theoritical optimal length")

# plt.xlabel("number of ants")
# plt.ylabel("length of path")
# plt.title("bays29 evaporation rate evolution")
# plt.legend()
# plt.savefig('bays29_ant_nbrs_200iter.png')

########### Beta graphs #########"


# possible_betas = np.arange(1,11,1)
# res = []
# graph = graph(distances)
# opt_path = reader.opt_reader("bays29.opt.tour")
# opt_length = graph.get_path_length(opt_path)

# for beta in possible_betas:
#     graph.iterations(5, EVAP_RATE, NBR_ANTS, Q, ALPHA, beta)
#     graph.get_ants_paths()
#     best_ant = graph.get_best_ant()
#     print(best_ant.get_final_distance_travelled())
#     res.append(best_ant.get_final_distance_travelled())

# plt.plot(possible_betas ,res , label="ants best length")
# plt.plot(possible_betas ,[opt_length]*len(possible_betas),label="theoritical optimal length")

# plt.xlabel("beta")
# plt.ylabel("length of path")
# plt.title("bays29 evaporation rate evolution")
# plt.legend()
# plt.savefig('bays29_beta.png')


#####Possible aphas #########
# possible_alphas = np.arange(1,11,1)
# res = []
# graph = graph(distances)
# opt_path = reader.opt_reader("bays29.opt.tour")
# opt_length = graph.get_path_length(opt_path)

# for alpha in possible_alphas:
#     graph.iterations(5, EVAP_RATE, NBR_ANTS, Q, alpha, BETA)
#     graph.get_ants_paths()
#     best_ant = graph.get_best_ant()
#     print(best_ant.get_final_distance_travelled())
#     res.append(best_ant.get_final_distance_travelled())

# plt.plot(possible_alphas ,res , label="ants best length")
# plt.plot(possible_alphas ,[opt_length]*len(possible_alphas),label="theoritical optimal length")
# plt.xlabel("alpha")
# plt.ylabel("length of path")
# plt.title("bays29 evaporation rate evolution")
# plt.legend()
# plt.savefig('bays29_alpha.png')

