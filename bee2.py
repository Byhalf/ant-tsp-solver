import tsp_reader as reader
from random import randint
import numpy as np
import copy

NBR_BEES = 500
ITERATIONS = 10
STAGES = 5
#S node to visit during one stage
NODES_TO_VISIT = 5
#A, RHO, THETA parameters set by the analyst 
A = 5
RHO = 2
THETA = 2
# B memory
B = 5


class graph:
    def __init__(self, distances):

        self.distances = np.array(distances)
        self.best_partial_paths = []
        self.stage_nbr = 1
        self.iter_nbr = 1
        self.hive_pos = -1


    def get_bees(self):
        return self.bees
    def get_bee_visits(self):
        return self.bee_visits
    def get_distances(self):
        return self.distances

    def make_new_bees(self, hive_pos):
        self.bees = []
        for _ in range(NBR_BEES):
            self.bees.append(bee(self.get_distances(),hive_pos))

    def do_first_stage(self, nodes_to_visit):
        
        # rdm =  randint(0,len(self.distances-1))
        # while(rdm == self.hive_pos):
        #     rdm = randint(0,len(self.distances)-1)

        self.make_new_bees(0)
        for _ in range(nodes_to_visit):
            self.move_bees()
        self.update_bee_visits()
        paths = Paths()
        for bee in self.bees:
            paths.add_path(bee)
        self.stage_nbr += 1
        return paths

        
    def do_more_stages(self,n,paths):
        for _ in range(n):
            paths.abandon_paths(self.stage_nbr,self.iter_nbr)
            probs = paths.get_probabilities_of_paths()
            unengaged_bees = len(self.bees) - paths.nbr_bees
            choices = np.random.choice(paths.get_paths(),unengaged_bees,p=probs,replace=True)
            for path in choices:
                self.update_path_visits(path)
            self.stage_nbr += 1


    def do_iterations(self, nbr_iterations, nbr_stages,nodes_to_visit):

        self.bee_visits = [] # Used for the memories of bees
        for _ in range (nbr_iterations+1): #+1 because indexing from 1, simpler to understand the maths
            self.bee_visits.append(np.zeros((len(distances),len(distances)),dtype=int))

        nbr_to_visit = nodes_to_visit
        increment_nbr_visits = len(distances)//nodes_to_visit
        for _ in range(nbr_iterations-1):
            self.stage_nbr = 1
            if(nbr_to_visit < nbr_iterations-1):
                paths = self.do_first_stage(nbr_to_visit)
            else:
                paths = self.do_first_stage(len(distances)-1)

            nbr_to_visit += increment_nbr_visits
            self.do_more_stages(nbr_stages-1,paths)
            self.iter_nbr += 1


        return paths.get_best_path().path
     

    #adds k visits
    def add_bee_visit(self,n,m,k=1):
        #iter nbr - 1 because started at 1
        self.bee_visits[self.iter_nbr][n][m] += k

    #I could update it as I move the bee, but will it mirror real life?
    def update_bee_visits(self):
        for bee in self.bees:
            visit = bee.visited_cities
            #-1 because [i+1]
            for i in range(len(visit)-1):
                n = visit[i]
                m = visit[i+1]
                self.add_bee_visit(n,m)

    def update_path_visits(self,path):
        for i in range(len(path.path)-1):
            n = path.path[i]
            m = path.path[i+1]
            self.add_bee_visit(n,m, path.nbr_bees)

    def move_bees(self):
        for bee in self.get_bees():
            choice = bee.choose_city(self.get_bee_visits(),self.iter_nbr)
            bee.move_to_city(choice)

    def get_path_length(self,path):
        distance_traveled = 0
        for i in range (len(path)-1):
            distance_traveled +=  int(self.distances[path[i]][path[i+1]])
        return distance_traveled



    
        
  
class bee:
    def __init__(self, distances,hive_position):
        self.nbr_of_cities = len(distances)
        self.distances = distances
        self.visited_cities = [hive_position]
        self.cities_to_visit = []
        self.distance_travelled = 0
        for i in range (0,self.nbr_of_cities):
            if(i!= hive_position):
                self.cities_to_visit.append(i)

    def get_current_city(self):
        return self.visited_cities[-1]
    def get_distance(self,i,j):
        return int(self.distances[i][j])
    def get_visited_cities(self):
        return self.visited_cities
    def get_distance_travelled(self):
        return self.distance_travelled

    def move_to_city(self, city):
        self.visited_cities.append(city)
        self.cities_to_visit.remove(city)
        self.distance_travelled += self.get_distance(self.visited_cities[-1],self.visited_cities[-2])

    def choose_city(self,bee_visits,iter_nbr):
        i = self.get_current_city()
        z = iter_nbr
        size = len(bee_visits[0])
        memories = np.zeros((size,size),dtype=int)
        r = max(z-B,0)
        for k in range(r,z):
            memories += bee_visits[k]

        probabilities = []
        for j in self.cities_to_visit:
            value_j = self.node_value(memories,i,j,z) 
            value_l = 0
            for l in self.cities_to_visit:
                value_l += self.node_value(memories,i,l,z)
            
            probabilities.append( value_j/value_l) 
        return np.random.choice(self.cities_to_visit, 1, p=probabilities)[0]
               
    def node_value(self,memories,i,j,z):
        memory = memories[i][j]
        if(memory != 0):
            res = np.exp(-A*self.get_distance(i,j)*z/memory)
            if(res != res):
                #this cheks for nan since NaN != NaN
                return 0
            else:
                return res
        else:
            return 1


class Paths:
    def __init__(self):
        self.paths = []
        self.shortest = 999999999999999
        self.longest = -1
        self.max_bees = 1
        self.min_bees = NBR_BEES +1
        self.nbr_bees = 0

    def add_path(self,bee):
        length = bee.get_distance_travelled()
        path = bee.get_visited_cities()

        self.nbr_bees += 1
        if(self.shortest>length):
            self.shortest = length
            self.paths.append(Path(bee))

        elif(self.longest<length):
            self.longest = length
            self.paths.append(Path(bee)) 
           
        else:
            for existing_path in self.paths:
                if(existing_path.path == path):
                    if(existing_path.increment_bees() > self.max_bees):
                        self.max_bees +=1
                    break
            self.paths.append(Path(bee))


    def abandon_paths(self,stage_nbr,iter_nbr):
        choices = [True,False]
        for path in self.paths:            
            p = np.exp(-( path.length - self.shortest)/stage_nbr*iter_nbr)
            if(np.random.choice(choices, 1, p=[p,1-p])[0] == False):
                self.nbr_bees -= 1
                if(path.decrement_bees()):
                    self.paths.remove(path)

    def get_paths(self):
        return self.paths

    def set_min_bees(self):
        for path in self.paths:
            if (path.get_bees() < self.min_bees):
                self.min_bees = path.get_bees()

    def get_min_bees(self):
        if(self.min_bees == NBR_BEES+1):
            self.set_min_bees()
            return self.min_bees

        else:
            return self.min_bees
        
        
    def get_max_bees(self):
        return self.max_bees

    def get_normalised_bees(self,nbr_bees):
        #what to do in case of division by 0?
        divisor = (self.get_max_bees() - self.get_min_bees())
        if(divisor==0):
            return 0.5
        return (nbr_bees - self.get_min_bees())/divisor

    def get_shortest_length(self):
        return self.shortest

    def get_longest_length(self):
        return self.longest

    def get_normalised_length(self,length):
        divisor = self.get_longest_length()-self.get_shortest_length()
        if(divisor == 0):
            return 0.5
        return (length - self.get_shortest_length())/divisor

    def set_path_values(self):
        self.sum_path_values = 0
        for path in self.paths:
            norm_bees = self.get_normalised_bees(path.nbr_bees)
            norm_length = self.get_normalised_length(path.length)
            path.path_value = np.exp(RHO * norm_bees - THETA* norm_length)
            self.sum_path_values += path.path_value
        

    def get_probabilities_of_paths(self):
        self.set_path_values()
        probs = []
        for path in self.paths:
            probs.append(path.path_value/self.sum_path_values)
        return probs

    def get_best_path(self):
        for path in self.paths:
            if(path.length == self.shortest):
                return path

        

class Path:
    
    def __init__(self,bee):
        self.path = bee.get_visited_cities()
        self.length = bee.get_distance_travelled()
        self.nbr_bees = 1
        self.path_value = 0

    def get_bees(self):
        return self.nbr_bees
    def increment_bees(self):
        self.nbr_bees += 1
        return self.nbr_bees
    
    def decrement_bees(self):
        self.nbr_bees -= 1
        if(self.nbr_bees==0):
            return True
        return False

    










distances = reader.tsp_reader("bays29.tsp")

graph = graph(distances)
#paths = graph.do_first_stage()
#graph.do_more_stages(2,paths)
solution = graph.do_iterations(ITERATIONS,STAGES,NODES_TO_VISIT)




file =  open("res_bees_unmoving_hive.txt", "w")
opt_path = reader.opt_reader("bays29.opt.tour")
opt_path_length = graph.get_path_length(opt_path)
best_path_found = solution
best_distance_travelled = graph.get_path_length(solution)

file.write("opt path="+ str(opt_path) + "\n" + "opt length= "+ str(opt_path_length)
            +"\n"+ "best path found= " + str(best_path_found) +"\n"+
            "best distance travelled= " + str(best_distance_travelled))



