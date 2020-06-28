from random import randint, random 
from math import inf
from cmath import sqrt
import matplotlib.pyplot as plt 
import numpy as np



## variables
inp_file = "inputs.txt"
out_file = "outputs.txt"

inputs = [int(x[:-1]) for x in open(inp_file, "r").readlines()]
outputs = [int(x[:-1]) for x in open(out_file, "r").readlines()]


## objects
'''
class Indicate(object):
	"""docstring for indicator"""
	def __init__(self, input_list):
		self.input_list = input_list
'''
class Polynomial(object):
	"""docstring for Polynomial:
	   coeffs are from big degree to small degree and should be a list:
		    -coeff = [9, 8, 7, 6, 5, 4, 3, 2, 1]
	   all degrees should be counted

	"""
	def __init__(self, coeffs):
		self.coeffs = list(coeffs);
		i = 0
		#print(self.coeffs)
		while self.coeffs[i] == 0 and len(self.coeffs) != 1:
			#print(self.coeffs)
			self.coeffs.pop(0)

		self.coeffs = tuple(self.coeffs)
		


	def __call__(self, x):
		a = [i for i in range(len(self.coeffs))]
		a.reverse()
		return sum([coeff * x ** i for coeff, x, i in zip(self.coeffs, len(self.coeffs)*[x], a)])

	def show(self):
		string = ""
		power = len(self.coeffs) - 1
		for c in self.coeffs:
			string += "(%d * x**%d )"%(c, power)
			if power != 0:
				string += " + "
			power -= 1
		return string

	def __add__(self, other):
		"""
		adding two polynomials be like:
			- (8, 7, 6) + (-11, 56, -901)

		"""
		other_coeffs = list(other.coeffs[:])
		while len(other_coeffs) < len(self.coeffs):
			other_coeffs.insert(0, 0)
		Sum = (x + y for x, y in zip(self.coeffs, other_coeffs))
		return Polynomial(Sum)

	def __mul__(self, other):
		"""
		multiplying two polynomials be like:
			- (8, 7, 6) * (-11, 56, -901)

		"""
		a = list(zip(self.coeffs[:], [i for i in range(len(self.coeffs)-1, -1, -1)]))
		b =list(zip(other.coeffs[:], [i for i in range(len(other.coeffs)-1, -1, -1)]))
		new = []

		for a_i in a:

			for b_j in b:

				new += [(a_i[0] * b_j[0], a_i[1] - b_j[1])]


		has_swapped = True

		num_of_iterations = 0

		while(has_swapped):
			has_swapped = False
			for i in range(len(new) - num_of_iterations - 1):
				if new[i][1] > new[i+1][1]:
	                # Swap
					new[i], new[i+1] = new[i+1], new[i]
					has_swapped = True
			num_of_iterations += 1
		
		final = [number for number, power in new]
		print("multiplying of %s and %s : "%(self.show(), other.show()), Polynomial(final).show())
		return Polynomial(final)

	def __div__(self, other):
		p = Polynomial(self.coeffs)
		q = Polynomial(other.coeffs)



	def __sub__(self, other):

		other_coeffs = list(other.coeffs[:])
		while len(other_coeffs) < len(self.coeffs):
			other_coeffs.insert(0, 0)
		sub = (x - y for x, y in zip(self.coeffs, other_coeffs))
		return Polynomial(sub)

	def deg(self):
		if sum(tuple(map(abs, self.coeffs))):
			return len(self.coeffs)-1

		else:
			raise ValueError("deg cannot be defined for zero polynomial")
	def roots(self):
		degree = self.deg()

		def degree_1():
			return ((- self.coeffs[-1]) / (self.coeffs[0]), ) 

		def degree_2():
			a = self.coeffs[0] 
			b = self.coeffs[1]
			c = self.coeffs[2]

			delta = (b ** 2) - (4 * a * c)

			#if delta < 0:
			#	return None

			p_1 = (-b + sqrt(delta)) / (2 * a)

			p_2 = (-b - sqrt(delta)) / (2 * a)

			return (p_1, p_2)

		def degree_3():
			a = self.coeffs[0] ; b = self.coeffs[1]
			c = self.coeffs[2] ; d = self.coeffs[3]

			alpha = ((- b**3 ) / ( 27 * a ** 3)) + ((b * c) / (6 * a ** 2)) - (d / (2 * a))
			beta = (c / (3 * a)) - (b ** 2 / (9 * (a ** 2)))
			gamma = b / (3 * a)

			p_0 = sqrt(alpha ** 2 + beta ** 3)

			p_1 = (alpha + p_0) ** (1 / 3)

			p_2 = (alpha - p_0) ** (1 / 3)

			return (p_1 + p_2 - gamma, );

		def degree_4():
			a = self.coeffs[0] ; b = self.coeffs[1]
			c = self.coeffs[2] ; d = self.coeffs[3]
			e = self.coeffs[4]

			p_1 = (2*c**3) - (9*b*c*d) + (27*a*(d**2)) + (27*(b**2)*e) - (72*a*c*e)

			p_2 = p_1 + sqrt((-4)*((c**2)-(3*b*d)+(12*a*e))**3+p_1**2)

			p_3 = ((c**2)-(3*b*d)+(12*a*e))/((3*a)*((p_2/2)**(1/3))) + ((p_2/2)**(1/3))/(3*a)

			p_4 = sqrt(((b**2)/(4*a**2)) - ((2*c)/(3*a)) + p_3)

			p_5 = ((b**2)/(2*a**2)) - ((4*c)/(3*a)) - p_3

			p_6 = ((-b**3/a**3) + ((4*b*c)/(a**2)) - ((8*d)/a) ) / (4*p_4)

			alpha = -(b / (4 * a))
			beta = p_4 / 2
			gamma = sqrt(p_5 - p_6)/2
			delta = sqrt(p_5 + p_6)/2

			x_1 = alpha - beta - gamma
			x_2 = alpha - beta + gamma
			x_3 = alpha + beta + delta
			x_4 = alpha + beta - delta

			return (x_1, x_2, x_3, x_4);


		if degree == 1:
			roots = degree_1()

		elif degree == 2:
			roots = degree_2()

		elif degree == 3:
			roots = degree_3()

		elif degree == 4:
			roots = degree_4()

		else:

			raise ValueError("equation or polynomials from degree 5 or more are not supported")
		return roots

class Equation(object):
	def __init__(self, p1, p2):
		self.p = p1 - p2
		self.answer = self.p.roots()

'''
class Function(object):
	"""docstring for Function"""
	def __init__(self, function):
		self.function = function

	def __call__(self, x):
		return self.function(x)

	def __add__(self, other):
		return Function(lambda x : self.function(x) + other.function(x))

	def __mul__(self, other):
		return Function(lambda x : self.function(x) * other.function(x))

	def __sub__(self, other):
		return Function(lambda x : self.function(x) - other.function(x))

	def __div__(self, other):
		return Function(lambda x : self.function(x) / other.function(x))
'''



def Indicator(inputs):
	r = randint(1, 1000) + random()
	print("inputs : ", inputs)
	indicators = [Polynomial([1, -a]) for a in inputs]
	#for i in indicators:
	#	print(i.show())
	indicator = Polynomial([1])

	for pl in indicators:
		print("polynomial : ", pl.show())
		indicator *= pl
	#print(indicator.show())
	return indicator

def creator(inputs, outputs):
	inp = inputs[:]
	out = outputs[:]
	main = Polynomial([0])

	def create_one_sentence(index):
		inp_1 = inp[:]
		inp_1.pop(index)
		pl = Polynomial([1])
		for x in inp_1:
			pl_i = Polynomial([1, x])
			pl_2 = Polynomial([inp[index] - x])
			pl_3 = pl_i.div(pl_2)

			pl *= pl_3

		return pl_3 * Polynomial([out[index]])

	for x in range(len(inp)):
		main += create_one_sentence(x)

	return main

def plot(p):
	x = np.linspace(-1, 1, 100)
	f = p(x)

	plt.scatter(x, f)
	plt.show()

plot(Polynomial([-0.8, 2.3, 0.5, 1, 0.2]))
