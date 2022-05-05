#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm

# In[42]:


from CNF_Creator import *
import numpy as np
import random
import time
import matplotlib.pyplot as plt
random.seed(1)


# In[376]:


class State:
    
    assignment = []
    fitness=0
    
    def __init__(self, assgn, cnf):
        self.assignment=assgn
        self.fitness = self.__fitness_fun(cnf)
    
    def __fitness_fun(self, cnf):
        return np.sum([(np.sign(i[0])*self.assignment[abs(i[0])]+np.sign(i[1])*self.assignment[abs(i[1])]+np.sign(i[2])*self.assignment[abs(i[2])]) > -3 for i in cnf])/len(cnf)


# In[386]:


class GeneticAlgorithm:
    
    population = [] #array of states
    
    
    def __init__(self, pop_size=20):
        self.pop_size=pop_size
        self.best_solution=None
        
    
    def generate_random_state(self, cnf, num_symbols=50):
        return State(assgn= [ -1 if i == False else 1 for i in (np.random.randint(-5, 5, num_symbols+1) < 0) ], cnf=cnf)
    
    def combination(self, parent1, parent2, cnf):
        rand_index= random.randint(1, len(parent1.assignment)-1)
        new_assgn=parent1.assignment[0:rand_index+1]
        new_assgn.extend(parent2.assignment[rand_index+1:len(parent2.assignment)])
        child = State(assgn=new_assgn, cnf=cnf)
        return child
    
    def mutation(self, state, prob=0.1):
        rand_index=random.randint(1, len(state.assignment)-1)
        if np.random.uniform() <= prob:
            state.assignment[rand_index]=state.assignment[rand_index]*-1
    
    def optimize(self,cnf, num_symbols=50, timeoutSeconds=45):
        generations = 0
        
        for i in range(self.pop_size):
            self.population.append(self.generate_random_state(cnf=cnf, num_symbols=num_symbols))
            
        timeout= time.time()+timeoutSeconds
        
        while True:
            for i in self.population:
                if self.best_solution==None or self.best_solution.fitness<i.fitness:
                    self.best_solution = i
            print (f"Best soln for Generation {generations}: {self.best_solution.fitness}")
            if time.time() > timeout or (max([i.fitness for i in self.population])==1):
                break
                
            sum_fitness=(np.sum([i.fitness for i in self.population])) 
            prob_array=[i.fitness for i in self.population]/sum_fitness
            new_population = []
            for i in range(self.pop_size):
                
                parent1=np.random.choice(a=self.population, p=prob_array)
                parent2=np.random.choice(a=self.population, p=prob_array)
                
                child = self.combination (parent1, parent2, cnf)
                self.mutation(state=child)
                new_population.append(child)
                                
            self.population = new_population
            generations=generations+1
            
        return self.population       


# In[378]:


class ImprovedGeneticAlgorithm: 
    population = [] #array of states
    
    def __init__(self, pop_size=50):
        self.pop_size=pop_size
        self.best_solution=None
        
    def generate_random_state(self, cnf, num_symbols=50):
        return State(assgn= [ -1 if i == False else 1 for i in (np.random.randint(-5, 5, num_symbols+1) < 0) ], cnf=cnf)
        
    def combination(self, parent1, parent2,cnf):
        rand_index= random.randint(1, len(parent1.assignment)-1)
        new_assgn1=parent1.assignment[0:rand_index+1]
        new_assgn1.extend(parent2.assignment[rand_index+1:len(parent2.assignment)])
        child1 = State(assgn=new_assgn1, cnf=cnf)
        
        new_assgn2=parent2.assignment[0:rand_index+1]
        new_assgn2.extend(parent1.assignment[rand_index+1:len(parent1.assignment)])
        child2= State(assgn= new_assgn2, cnf=cnf)
        #return child1
        return child1 if child1.fitness>child2.fitness else child2
    
    def mutation(self, state, prob=0.1):
        rand_index=random.randint(1, len(state.assignment)-1)
        if np.random.uniform() <= prob:
            state.assignment[rand_index]=state.assignment[rand_index]*-1
            
    def elitism(self, new_population):
        total_population=self.population+new_population
        total_population.sort(key= lambda x: x.fitness, reverse = True)
        self.population = total_population[0:min(self.pop_size, len(total_population))]
    
    def cull(self, threshold=0.0199):
        prob_array=[i.fitness for i in self.population]/(np.sum([i.fitness for i in self.population])) 
        #print (prob_array)
        for i in range(len(prob_array)): 
            if prob_array[i] < threshold:
                self.population[i].fitness=0
        
    
    
    def optimize(self, cnf, num_symbols= 50, timeoutSeconds=45, elitism = False, culling= False, softmax_activation= False):
        generations = 0
        
        for i in range(self.pop_size):
            self.population.append(self.generate_random_state(cnf=cnf, num_symbols=num_symbols))
            
        timeout= time.time()+timeoutSeconds
        
        while True:
            for i in self.population:
                if self.best_solution==None or self.best_solution.fitness<i.fitness:
                    self.best_solution = i
            #print (f"Best soln for Generation {generations}: {self.best_solution.fitness}")
            if time.time() > timeout or (max([i.fitness for i in self.population])==1):
                break
                
            sum_fitness=(np.sum([i.fitness for i in self.population])) 
            prob_array=[i.fitness for i in self.population]/sum_fitness
            
            if softmax_activation:
                sum_probs=np.sum([np.exp(i) for i in prob_array if i!=0])
                prob_array=[np.exp(i)/sum_probs if i!=0 else 0 for i in prob_array]
                
            new_population = []
            for i in range(self.pop_size):
                
                parent1=np.random.choice(a=self.population, p=prob_array)
                parent2=np.random.choice(a=self.population, p=prob_array)
                
                child = self.combination (parent1, parent2, cnf)
                self.mutation(state=child)
                new_population.append(child)
            
            if elitism:
                self.elitism(new_population)
            else:
                self.population = new_population
            
            
            if culling:
                self.cull(threshold=0.0198)
                
            generations=generations+1
            
        return self.population       


# In[387]:


cnfCreator=CNF_Creator(n=50)
sentence=cnfCreator.ReadCNFfromCSVfile()


# In[390]:


model=GeneticAlgorithm()
population=model.optimize(num_symbols=50,cnf=sentence, timeoutSeconds=45)


# In[257]:


model=ImprovedGeneticAlgorithm(pop_size=50)
population=model.optimize(num_symbols=50,cnf=sentence, timeoutSeconds=45, culling=True)


# In[163]:


print(max([i.fitness for i in population]))


# In[150]:


print(model.best_solution.fitness)


# Best improvement

# In[178]:


random.seed(1)
best_fitness=[]
for i in range(100,300,20):
    currSum=0
    for j in range (10):
        cnfCreator=CNF_Creator(n=50)
        sentence=cnfCreator.CreateRandomSentence(m=i)
        model=ImprovedGeneticAlgorithm(pop_size=100)
        population=model.optimize(num_symbols=50,cnf=sentence, timeoutSeconds=20)
        currSum+=model.best_solution.fitness
    best_fitness.append(currSum/10)
    print(f"{i} clauses: {best_fitness[-1]}")


# In[181]:


plt.ylim(bottom=0.95)
plt.plot(range(100,300,20), best_fitness)


# In[133]:


random.seed(1)
best_fitness=[]
for i in range(100,300,20):
    currSum=0
    for j in range (10):
        cnfCreator=CNF_Creator(n=50)
        sentence=cnfCreator.CreateRandomSentence(m=i)
        model=GeneticAlgorithm()
        population=model.optimize(num_symbols=50,cnf=sentence, timeoutSeconds=20)
        currSum+=model.best_solution.fitness
    best_fitness.append(currSum/10)
    print(f"{i} clauses: {best_fitness[-1]}")


# # Testing with pure numpy

# In[391]:


class State:

    fitness=0
    
    def __init__(self, assgn, cnf):
        self.assignment=assgn
        self.fitness = self.__fitness_fun(cnf)
    
    def __fitness_fun(self, cnf):
        return np.sum([(np.sign(i[0])*self.assignment[abs(i[0])]+np.sign(i[1])*self.assignment[abs(i[1])]+np.sign(i[2])*self.assignment[abs(i[2])]) > -3 for i in cnf])/len(cnf)


# In[392]:


class GeneticAlgorithm:
    
    population = [] #array of states
    
    
    def __init__(self, pop_size=20):
        self.pop_size=pop_size
        self.best_solution=None
        
    
    def generate_random_state(self, cnf, num_symbols=50):
        assign=np.random.randint(0,2, size= num_symbols+1)
        return State(assgn=np.where(assign>0,1,-1), cnf=cnf)
    
    def combination(self, parent1, parent2, cnf):
        rand_index= random.randint(1, len(parent1.assignment)-1)
        child = State(assgn=np.concatenate((parent1.assignment[0:rand_index+1],parent2.assignment[rand_index+1:len(parent2.assignment)])), cnf=cnf)
        return child
    
    def mutation(self, state, prob=0.1):
        rand_index=random.randint(1, len(state.assignment)-1)
        if np.random.uniform() <= prob:
            state.assignment[rand_index]=state.assignment[rand_index]*-1
    
    def optimize(self,cnf, num_symbols=50, timeoutSeconds=45):
        generations = 0
        
        for i in range(self.pop_size):
            self.population.append(self.generate_random_state(cnf=cnf, num_symbols=num_symbols))
            
        timeout= time.time()+timeoutSeconds
        
        while True:
            for i in self.population:
                if self.best_solution==None or self.best_solution.fitness<i.fitness:
                    self.best_solution = i
            print (f"Best soln for Generation {generations}: {self.best_solution.fitness}")
            if time.time() > timeout or (max([i.fitness for i in self.population])==1):
                break
                
            sum_fitness=(np.sum([i.fitness for i in self.population])) 
            prob_array=[i.fitness for i in self.population]/sum_fitness
            new_population = []
            for i in range(self.pop_size):
                
                parent1=np.random.choice(a=self.population, p=prob_array)
                parent2=np.random.choice(a=self.population, p=prob_array)
                
                child = self.combination (parent1, parent2, cnf)
                self.mutation(state=child)
                new_population.append(child)
                                
            self.population = new_population
            generations=generations+1
            
        return self.population       


# In[393]:


cnfCreator=CNF_Creator(n=50)
sentence=cnfCreator.ReadCNFfromCSVfile()


# In[394]:


model=GeneticAlgorithm()
population=model.optimize(num_symbols=50,cnf=sentence, timeoutSeconds=45)


