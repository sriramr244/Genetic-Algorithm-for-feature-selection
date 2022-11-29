# -*- coding: utf-8 -*-
"""Genetic_Algo_new_version.pynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_uSUh0mgQnoZXi4XPKHczIkmFt9w06uE

Author: Sriram Ranganathan, Meng. ECE, University of Waterloo
(Main Code)
"""

import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm 
import sys
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy.random as npr

def estimate(T_x, T_y, V_x, V_y, clf_type):
  if clf_type == "SVM":
    clf = svm.SVC(C = 1, kernel = 'poly', gamma = 'auto')
    clf.fit(T_x, T_y)
    return clf.score(V_x, V_y)
  
  if clf_type == "RFC":
    clf_ww = RandomForestClassifier(max_depth=50, n_estimators= 200, random_state=0)
    clf_ww.fit(T_x, T_y)
    return clf_ww.score(V_x, V_y)

def roulette_select_one(c_population):
  max = sum([f.fitness for f in c_population])
  probs = [f.fitness/max for f in c_population]
  return c_population[npr.choice(len(c_population), p=probs)]

#Code author: S
class candidate:
  def __init__(self, bitstream,  fitness = 0.00):
    self.fitness = fitness
    self.bitstream = bitstream
  def __eq__(self, x):
    if self.bitstream == x.bitstream:
      return True
    return False

class top_solution:
  def __init__(self, new,  iter = 0):
    self.iter = iter
    self.new = new
class Genetic_algorithm:

  def __init__(self, input_data_x, input_data_y, max_population, crossover_prob, mutation_r, stop_by_f, stop_fitness, fitness_func):
    self.input_data_x = input_data_x
    self.input_data_y = input_data_y.to_numpy()
    #self.input_data_v_x = input_data_v_x
    #self.input_data_v_y = input_data_v_y.to_numpy()
    self.max_population = max_population
    self.columns = self.input_data_x.columns
    self.fitness_func = fitness_func
    self.populate()
    self.calculate_fitness()
    self.mating_pool_size = max_population//5
    self.crossover_prob = crossover_prob
    self.mutation_r = mutation_r
    self.Best_solutions = []
    self.Best_solutions_bit = []
    self.Best_iteration = 0 
    self.stop_fitness = stop_fitness
    self.stop_by_f = stop_by_f
    self.fitness_dispersion = []
    self.len_bitstream_dispersion = []
  def evolve(self, no_iters):
    print("Genetic Algorithm Evolving")
    i = 0
    self.average = []
    self.Top_sols = []
    self.worst_sols = []
    self.tot_crossov = []
    self.tot_mut = []
    l = range(0,no_iters)
    self.crossover = 0
    self.mutation = 0
    for i in l:

      top_sol = self.current_population[0]
      self.Best_solutions.append(top_sol.fitness)
      self.Best_solutions_bit.append(top_sol.bitstream)
      print("Top solution fitness "+ str(top_sol.fitness))
      print("Iteration_No: ", i)
      
      fitness = [m.fitness for m in self.current_population]
      self.average.append(sum(fitness)/len(fitness))
      self.Top_sols.append(max(fitness))
      self.worst_sols.append(min(fitness))
      if ((top_sol.fitness > self.stop_fitness) & (self.stop_by_f)):
        return top_sol
      self.current_population = self.cross_over_mutate(self.current_population)
      self.calculate_fitness()
      i+=1
      self.current_population.sort(key=lambda x: x.fitness, reverse=True)

    #self.Best_solutions.sort(key=lambda x: x.fitness, reverse=True)
    #print("Best solution fitness: ", self.Best_solutions[0].fitness)
    #print("Best Generation: ", self.Best_solutions)
      self.tot_crossov.append(self.crossover)
      self.tot_mut.append(self.mutation)
    return top_sol


  def populate(self, initial = False):
    print("Creating Initial population")
    self.current_population = []
    for i in range(0,self.max_population):
      bitstream = []
      for i in self.input_data_x.columns:
        if random.randrange(10)<=5:
          bitstream.append(1)
        else:
          bitstream.append(0)
      
      new_cand = candidate(bitstream)
      rep = False
      for i in self.current_population:
        if bitstream == i.bitstream:
          rep = True
          break
      if rep == True:
        continue
      self.current_population.append(new_cand)
    return

  def calculate_fitness(self):
    print("Calculating fitness")
    
    for i in self.current_population:
      new_data_frame = self.input_data_x
      bitstream = i.bitstream
      drop_columns = []
      for k in range(0,len(bitstream)):
        if bitstream[k] == 1:
          continue
        if bitstream[k] == 0:
          drop_columns.append(self.columns[k])
        
      new_data_frame = self.input_data_x.drop(drop_columns, axis = 1)
      Train_x = new_data_frame.to_numpy()
      '''new_data_frame = self.input_data_v_x.drop(drop_columns, axis = 1)
      Test_x = new_data_frame.to_numpy()'''
      X_train, X_test, y_train, y_test = train_test_split(Train_x, self.input_data_y, test_size=0.2)
      i.fitness = estimate(X_train, y_train, X_test, y_test, self.fitness_func)
    return
  

  def cross_over_mutate(self, current_population):
    self.fitness_dispersion.append([f.fitness for f in self.current_population])
    y = [np.array(f.bitstream) for f in self.current_population]
    y = [np.sum(l) for l in y]
    self.len_bitstream_dispersion.append(y)
    current_population.sort(key=lambda x: x.fitness, reverse=True)
    new_population = current_population[0:2]
    print("Top 2 Fitness of new population", new_population[0].fitness, new_population[1].fitness)
    mating_pool = current_population[:self.mating_pool_size].copy()
    m = 0
    while(len(new_population)<len(current_population)):
      n = m
      if m>=len(mating_pool):
        n = m%len(mating_pool)
      p1 = roulette_select_one(mating_pool)
      new_mating_pool = mating_pool.copy()
      new_mating_pool.pop(n)
      new_cand = candidate([], 0)
      if random.uniform(0, 1)<=self.crossover_prob:
        self.crossover+=1
        p2 = roulette_select_one(mating_pool)
        trait_split = random.randrange(self.input_data_x.shape[1])
        L = [k for k in range(trait_split,self.input_data_x.shape[1])]
        trait_split1 = random.choice(L)
        new_bitstream = self.mutate(p1.bitstream[0:trait_split] + p1.bitstream[trait_split:trait_split1]+p2.bitstream[trait_split1:])
        new_cand.bitstream = new_bitstream
        bs = [str(k) for k in new_cand.bitstream]
        rep = False
        '''
        for u in new_population:
          if new_cand.bitstream == u.bitstream:
            rep = True
        if rep:
          continue
        '''
        new_population.append(new_cand)
      m+=1
    current_population = new_population.copy()
    
    current_population.sort(key=lambda x: x.fitness, reverse=True)
    return current_population
  

  def mutate(self, bitstream):
    for i in range(0,len(bitstream)):
      if random.uniform(0, 1)<=self.mutation_r:
        self.mutation
        if bitstream[i]==0:
          bitstream[i] = 1
        else:
          bitstream[i] = 0
    return bitstream

new_df = pd.read_csv('/content/drive/MyDrive/ECE_750_data/C_A_D_cleaned.csv')

new_df.head()

Train = new_df
#Vaidate = new_df.drop(Train.index)

Train_data_x = Train.drop(['Out'], axis = 1)
#Test_data_x = Vaidate.drop(['Out'], axis = 1)

Train_data_y = Train['Out']
#Test_data_y = Vaidate['Out']

GA_CAD = Genetic_algorithm(input_data_x = Train_data_x,input_data_y = Train_data_y,
                           max_population = 100, crossover_prob = 0.9, 
                           mutation_r = 0.01, stop_by_f = False, stop_fitness = 0.81, fitness_func="SVM")

GA_CAD.evolve(100)

ind = np.argmax(GA_CAD.Best_solutions)
ind

max(GA_CAD.Best_solutions)

best_bit = GA_CAD.Best_solutions_bit[ind]

x = Train_data_x.columns
drop = []
for i in range(0, len(best_bit)):
  if best_bit[i] == 1:

    continue
  else:
    drop.append(x[i])
print("No. of columns to be dropped: ",len(drop))
print(drop)

#plt.plot(range(0,len(GA_CAD.Top_sols)),GA_CAD.average, label = "Avg Fitness")
plt.plot(range(0,len(GA_CAD.Top_sols)),GA_CAD.Top_sols, label = "Max Fitness")
#plt.plot(range(0,len(GA_CAD.Top_sols)),GA_CAD.worst_sols, label = "Min Fitness")
#plt.ylim(0.4, 0.9)
plt.xlabel('Generations') 
plt.ylabel('Validation Accuracy from solutions(fitness)') 
plt.legend(loc="lower right")

# displaying the title
plt.title("Cardiac Arrest Dataset Size = 300")

for i in range(0, len(GA_CAD.len_bitstream_dispersion)):
  plt.scatter(GA_CAD.len_bitstream_dispersion[i],GA_CAD.fitness_dispersion[i])
  plt.title("Generation "+ str(i))
  plt.xlabel("Length of bitsream")
  plt.ylabel("Fitness")
  plt.ylim(0.5, 0.85)
  plt.xlim(120, 180)
  plt.savefig("/content/drive/MyDrive/ECE_750_data/GEN/Gen_"+str(i)+".jpg")
  plt.close()

plt.scatter(GA_CAD.len_bitstream_dispersion[99],GA_CAD.fitness_dispersion[99], label = 'Gen 100')
plt.scatter(GA_CAD.len_bitstream_dispersion[49],GA_CAD.fitness_dispersion[49], label = 'Gen 50')
plt.scatter(GA_CAD.len_bitstream_dispersion[0],GA_CAD.fitness_dispersion[0], label = 'Gen 1')
plt.title("Different Generation of solutions")
plt.xlabel("Length of bitsream")
plt.ylabel("Fitness")
plt.legend(loc="best")

GA_CAD.fitness_dispersion[0]

"""## White Wine Quality Data set"""

df_white_wine = pd.read_csv('/content/drive/MyDrive/ECE_750_data/winequality-white.csv')

df_white_wine.head()

Train_x_ww = df_white_wine.drop(['quality'], axis =1)
Train_y_ww = df_white_wine.quality

GA_WW = Genetic_algorithm(input_data_x = Train_x_ww,input_data_y = Train_y_ww,
                           max_population = 20, crossover_prob = 0.9, 
                           mutation_r = 0.01, stop_by_f = True, stop_fitness = 0.81, fitness_func="RFC")

GA_WW.evolve(20)

ind_ww = np.argmax(GA_WW.Best_solutions)
ind_ww

max(GA_WW.Best_solutions)

best_bit_ww = GA_WW.Best_solutions_bit[ind_ww]
best_bit_ww

x_ww = df_white_wine.drop(['quality'], axis =1).columns
drop_ww = []
for i in range(0, len(best_bit_ww)):
  if best_bit_ww[i] == 1:
    continue
  else:
    drop_ww.append(x_ww[i])
print("No. of columns to be dropped: ",len(drop_ww))
print(drop_ww)

plt.plot(range(0,len(GA_WW.Top_sols)),GA_WW.average, label = "Avg Fitness")
plt.plot(range(0,len(GA_WW.Top_sols)),GA_WW.Top_sols, label = "Max Fitness")
plt.plot(range(0,len(GA_WW.Top_sols)),GA_WW.worst_sols, label = "Min Fitness")
plt.ylim(0.55, 0.8)
plt.xlabel('Generations') 
plt.ylabel('Validation Accuracy from solutions(fitness)') 
plt.legend(loc="lower right")

# displaying the title
plt.title("White wine")

"""### GA for Breast cancer"""

df_BC = pd.read_csv('/content/drive/MyDrive/ECE_750_data/B_C_cleaned.csv')

df_BC.head()

Train_x_bc = df_BC.drop(['Out'], axis =1)
Train_y_bc = df_BC.Out

GA_BC = Genetic_algorithm(input_data_x = Train_x_bc,input_data_y = Train_y_bc,
                           max_population = 20, crossover_prob = 0.9, 
                           mutation_r = 0.01, stop_by_f = False, stop_fitness = 0.98, fitness_func="RFC")

GA_BC.evolve(100)

ind_BC = np.argmax(GA_BC.Best_solutions)
ind_BC

max(GA_BC.Best_solutions)

best_bit_BC = GA_BC.Best_solutions_bit[ind]

x_BC = Train_x_bc.columns
drop_BC = []
for i in range(0, len(best_bit_BC)):
  if best_bit_BC[i] == 1:
    continue
  else:
    drop_BC.append(x_BC[i])
print("No. of columns to be dropped: ",len(drop_BC))
print(drop_BC)

plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.average, label = "Avg Fitness")
plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.Top_sols, label = "Max Fitness")
plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.worst_sols, label = "Min Fitness")
plt.ylim(0.8, 1.05)
plt.xlabel('Generations') 
plt.ylabel('Validation Accuracy from solutions(fitness)') 
plt.legend(loc="lower right")

# displaying the title
plt.title("Breast cancer")

plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.average, label = "line 1")
plt.xlabel('Generations') 
plt.ylabel('Avg Validation Accuracy from solutions(fitness)') 
  
# displaying the title
plt.title("Avg fitness vs Generations")

plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.Top_sols, label = "line 1")
plt.xlabel('Generations') 
plt.ylabel('Max Validation Accuracy from solutions(fitness)') 
  
# displaying the title
plt.title("Max fitness vs Generations")

plt.plot(range(0,len(GA_BC.Top_sols)),GA_BC.worst_sols, label = "line 1")
plt.xlabel('Generations') 
plt.ylabel('Min Validation Accuracy from solutions(fitness)') 
  
# displaying the title
plt.title("Min fitness vs Generations")
