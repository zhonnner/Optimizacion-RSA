import random as rnd
import math
import time
import numpy as np

print("  -->  Aplicando Nodo y Arco consistencia  <--")
domains = {
    'x1':  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'x2':  [0,1,2,3,4,5,6,7,8,9,10],
    'x3':  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    'x4':  [0,1,2,3,4],
    'x5':  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    'y' :  [0,1],
}

constraints = {
    ('x1', 'x2'): lambda a, b: a <= 20 - b,
    ('x2', 'x1'): lambda b, a: 20 - b >= a,
    
    ('x1', 'x2'): lambda a, b: a * 150 <= 1800 - b * 300,
    ('x2', 'x1'): lambda b, a:  1800 - b * 300 >= a * 150
}

def revise(x, y):
    revised = False
    x_domain = domains[x]
    y_domain = domains[y]
    all_constraints = [
        constraint for constraint in constraints if constraint[0] == x and constraint[1] == y]
    for x_value in x_domain:
        satisfies = False
        for y_value in y_domain:
            for constraint in all_constraints:
                constraint_func = constraints[constraint]
                if constraint_func(x_value, y_value):
                    satisfies = True
        if not satisfies:
            x_domain.remove(x_value)
            revised = True
    return revised

def ac3(arcs):
    queue = arcs[:]
    while queue:
        (x, y) = queue.pop(0)
        revised = revise(x, y)
        if revised:
            neighbors = [neighbor for neighbor in arcs if neighbor[1] == x]
            queue = queue + neighbors

arcs = [
    ('x1', 'x2'), ('x2', 'x1')
]
ac3(arcs)
for key, values in domains.items():
    print(f"  Dominio de {key}: {values}")


def checkConstraint(x):
		x1, x2, x3, x4, x5, y = x
		return (
			0 <= x1 <= 15
			and x2 <= 10*(1-y)
			and x3 <= 25*y
			and x4  <= 4
			and x5 <= 30
			
			and y <= 1
			
			and x1 * 1000 + x2 * 2000 + x3* 1500 + x4 * 2500 + x5 * 300 <= 50000 # x1 * 1000 <= 50000 - x2 * 2000
			and x1 + x2 <= 20 # x1 <= 20 - x2
			and 150*x1 + 300 * x2 <= 1800  #x1 * 150 <= 1800 - x2 * 300
            # and estado
		)
		
def isFeasible(x):
		return checkConstraint(x)		
		
class Problem:
	def __init__(self):
		self.dimension = 6


	def eval(self, x):
		# quality_points = [65, 90, 40, 60, 20]
		# costs = [150, 300, 40, 100, 10]
		x1, x2, x3, x4, x5, y = x
		f_min = 150*x1 +  300*x2*(1-y)  +  40*x3*y  +  100*x4  +  10*x5  # Función de minimización
		f_max =  65*x1 +   90*x2*(1-y)  +  40*x3*y  +   60*x4  +  20*x5  # Función de maximización
		
		w_min = 1  # Peso para la función de minimización
		w_max = 1  # Peso para la función de maximización
		
		f_obj = f_obj = w_max * f_max - w_min * f_min
		return (f_obj)


def ver(x):
	x1, x2, x3, x4, x5, y = x
	f_min = 150*x1 +  300*x2*(1-y)  +  40*x3*y  +  100*x4  +  10*x5  # Función de minimización
	f_max =  65*x1 +   90*x2*(1-y)  +  40*x3*y  +   60*x4  +  20*x5  # Función de maximización
	
	print("  f_min:", f_min)
	print("  f_max:", f_max)
	
def F_obj(x):
	x1, x2, x3, x4, x5, y = x
	f_min = 150*x1 +  300*x2*(1-y)  +  40*x3*y  +  100*x4  +  10*x5  # Función de minimización
	f_max =  65*x1 +   90*x2*(1-y)  +  40*x3*y  +   60*x4  +  20*x5  # Función de maximización
	
	# w_min = 0  # Peso para la función de minimización
	# w_max = 2  # Peso para la función de maximización
	#esta se debe cambiar para hacer la evaluacion de la/las funciones objetivos
	# f_obj = (0.5 * (x1/15) + 0.2 * (x2*y/10) + 0.1 * (x3*(1-y)/25) + 0.1 * (x4/4) + 0.1 * (x5/30)) - (0.4 * (x1/15) + 0.3 * (x2*y/10) + 0.1 * (x3*(1-y)/25) + 0.1 * (x4/4) + 0.1 * (x5/30))
	#f_obj =  f_max
	f_obj = (f_max / 2140) * 0.5 + ((3000 - f_min) / 3000 - 0) * 0.5
	#f_obj = ((100 * f_max) / 2620) + 100 - ((100 * f_min) / 3500)
	return (f_obj)

class Agent(Problem):
	def __init__(self):
		self.p = Problem()
		self.x = [] 
		# for _ in range(self.p.dimension):
		# print("->", domains)
		self.x.append(rnd.choice(domains['x1']))
		self.x.append(rnd.choice(domains['x2']))
		self.x.append(rnd.choice(domains['x3']))
		self.x.append(rnd.choice(domains['x4']))
		self.x.append(rnd.choice(domains['x5']))
		
		#y es 0 o 1
		self.x.append(rnd.choice(domains['y']))


	def isBetterThan(self, g):
		return self.fit() > g.fit()

	def fit(self):
		return self.p.eval(self.x)

	def move(self, g):
		# for i in range(self.p.dimension):
			# self.x[i] = self.toBinary(self.x[i] + rnd.uniform(0, 1) * (g.x[i] - self.x[i]))
		self.x[0] = (rnd.choice(domains['x1']))
		self.x[1] = (rnd.choice(domains['x2']))
		self.x[2] = (rnd.choice(domains['x3']))
		self.x[3] = (rnd.choice(domains['x4']))
		self.x[4] = (rnd.choice(domains['x5']))
		self.x[5] = self.toBinary(self.x[5] + rnd.uniform(domains['y']) * (g.x[5] - self.x[5]))
	
	def toBinary(self, x):
		return 1 if (1 / (1 + math.exp(-x))) > rnd.random() else 0

	def _str_(self) -> str:
		return f"  fit:{self.fit()} x:{self.x}"

	def copy(self, a):
		self.x = a.x.copy()
	
	def toString(self):
		return self.x

class Swarm:
	def __init__(self):
		self.maxIter = 500
		self.nAgents = 10
		self.swarm = []
		self.g = Agent()

	def getAgents(self):
		return self.nAgents
		
	def solve(self):
		self.initRand()
		self.evolve()

	def initRand(self):
		print("\n  -->  Fase de inicialización  <-- ")
		for _ in range(self.nAgents):
			while True:
				a = Agent()
				if isFeasible(a.toString()):
					break
			self.swarm.append(a)

		# self.g.copy(self.swarm[0])
		# for i in range(1, self.nAgents):
			# if self.swarm[i].isBetterThan(self.g):
				# self.g.copy(self.swarm[i])

		self.swarmToConsole()
		# self.bestToConsole()

	def evolve(self):
		N = self.nAgents  # Número de filas
		T = self.maxIter  # Valor de T
		Dim = 6  # Número de columnas
		
		UB = [12, 6, 25, 4, 30, 1]  # Valor superior del límite
		LB = 0  # Valor inferior del límite	
		print("\n  -->  Comienza RSA  <-- ")
		
		Best_P = [1 for i in range(N)]
		Best_F = - float('inf')
		X = np.empty((self.nAgents, 6), dtype=object)
		for i in range(self.nAgents):
			X[i] = self.swarm[i].toString()
			
		Xnew = np.zeros((N, Dim))
		Conv = np.zeros(T)
	
		t = 0
		Alpha = 0.1
		Beta = 0.005
		Ffun = np.zeros(X.shape[0])
		Ffun_new = np.zeros(X.shape[0])
		
		for i in range(X.shape[0]):
			
			Ffun[i] = F_obj(X[i, :])  # Calcular los valores de aptitud de las soluciones
			if Ffun[i] > Best_F:
				Best_F = Ffun[i]
				Best_P = X[i, :]

		# Imprimir el mejor valor de aptitud y la mejor solución
		print("  Mejor Fitnees:", Best_F)
		print("  Mejor solución hasta ahora:", Best_P)
		print()
		import sys
		while t < T:
			ES = 2 * np.random.randint(-1, 2) * (1 - (t / T))  # Ratio de probabilidad
			for i in range(1, X.shape[0]):
				for j in range(X.shape[1]):
					R = (Best_P[j] - X[np.random.randint(0, X.shape[0]), j]) / (Best_P[j] + sys.float_info.epsilon)
					P = Alpha + (X[i, j] - np.mean(X[i, :])) / (Best_P[j] * (UB[j] - LB) + sys.float_info.epsilon)
					Eta = Best_P[j] * P
					
					if t < T / 4:
						Xnew[i, j] = Best_P[j] - Eta * Beta - R * np.random.rand()
					elif t < 2 * T / 4 and t >= T / 4:
						Xnew[i, j] = Best_P[j] * X[np.random.randint(0, X.shape[0]), j] * ES * np.random.rand()
					elif t < 3 * T / 4 and t >= 2 * T / 4:
						Xnew[i, j] = Best_P[j] * P * np.random.rand()
					else:
						Xnew[i, j] = Best_P[j] - Eta * sys.float_info.epsilon - R * np.random.rand()
				
				# print("Xnew: -> ", Xnew[i, :])
				# print("UB: ->",UB)
				
				Flag_UB = Xnew[i, :] >= UB # verificar >=
				Flag_LB = Xnew[i, :] <= LB # verificar <=
				Xnew[i, :] = np.where(~(Flag_UB + Flag_LB), Xnew[i, :], Xnew[i, :] * (~(Flag_UB + Flag_LB)) + UB * Flag_UB + LB * Flag_LB) #
				Xnew[i, :] = np.floor(Xnew[i, :])
				# print(Xnew[i, :])
				# time.sleep(0.2)
				if(isFeasible(Xnew[i, :])):
					Ffun_new[i] = F_obj(Xnew[i, :])

					if Ffun_new[i] > Ffun[i]:
						X[i, :] = Xnew[i, :]
						Ffun[i] = Ffun_new[i]

					if Ffun[i] > Best_F:
						Best_F = Ffun[i]
						Best_P = X[i, :]

			Conv[t] = Best_F

			if t % 50 == 0:
				print(f"  En la iteración {t}, la mejor aptitud de la solución es {Best_F}")
			t += 1
		ver(Best_P)
		print("\n  La mejor solución obtenida por RSA es:", ", ".join([("X" if i!= 5 else "Y") + str(i+1) + "=" + str(int(Best_P[i])) for i in range(len(Best_P))]) )
		print("  El mejor valor óptimo de la función objetivo encontrado por RSA es:", Best_F)
				

	def swarmToConsole(self):
		# time.sleep(1)
		print("\n -- inicialización --")
		for i in range(self.nAgents):
			print(f"  {self.swarm[i].toString()}")

	def bestToConsole(self):
		print("\n -- Best --")
		print("X1  X2  X3  X4  X5 Y")
		print(f"{self.g.toString()}")

try:
	Swarm().solve()
except Exception as e:
	print(f"{e} \nCaused by {e._cause_}")
