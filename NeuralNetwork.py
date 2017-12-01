import math
import random
import numpy as np

class NeuralNetwork:
	def __init__(self, num_input, num_hidden, num_output):
		self.ni = num_input
		self.nh = num_hidden
		self.no = num_output
		
		self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
		self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
		self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)

		self.ihWeights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
		self.hoWeights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)

		self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
		self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)

		self.rnd = random.Random(10)
		self.initializeWeights()

	def initializeWeights(self):
		lo = 0.001 
		hi = 0.009
		for i in range(self.ni):
		  for j in range(self.nh):
		    self.ihWeights[i,j] = (hi - lo) * self.rnd.random() + lo
		
		for j in range(self.nh):
		  for k in range(self.no):
		    self.hoWeights[j,k] = (hi - lo) * self.rnd.random() + lo
			
		for j in range(self.nh):
		  self.hBiases[j] = (hi - lo) * self.rnd.random() + lo
		  
		for k in range(self.no):
		  self.oBiases[k] = (hi - lo) * self.rnd.random() + lo

	def train(self, x_values, t_values, maxEpochs, learnRate, crossError):
		hGrads = np.zeros(shape=[self.nh], dtype=np.float32)
		oGrads = np.zeros(shape=[self.no], dtype=np.float32)
		rand_indicies = np.arange(len(x_values))
		for epoch in range(maxEpochs):
			print("Epoch = ", epoch)
			if (self.crossEntropyError(x_values, t_values) < crossError):
				return;

			np.random.shuffle(rand_indicies)
			x_values = x_values[rand_indicies]
			t_values = t_values[rand_indicies]
			for i in range(len(x_values)):
				self.computeOutputs(x_values[i])
				self.computeGradient(t_values[i], oGrads, hGrads)
				self.updateWeightsAndBiases(learnRate, hGrads, oGrads)

	def computeOutputs(self, xValues):
		hSums = np.zeros(shape=[self.nh], dtype=np.float32)
		oSums = np.zeros(shape=[self.no], dtype=np.float32)

		for i in range(self.ni):
			self.iNodes[i] = xValues[i]

		for j in range(self.nh):
			for i in range(self.ni):
				hSums[j] += self.iNodes[i] * self.ihWeights[i,j]
			hSums[j] += self.hBiases[j]
		  
		for j in range(self.nh):
			self.hNodes[j] = self.hypertan(hSums[j])

		for k in range(self.no):
			for j in range(self.nh):
				oSums[k] += self.hNodes[j] * self.hoWeights[j,k]
			oSums[k] += self.oBiases[k]

		softOut = self.softmax(oSums)
		for k in range(self.no):
			self.oNodes[k] = softOut[k]
		return self.oNodes
		
	def computeGradient(self, t_values, oGrads, hGrads):
		for i in range(self.no):
			oGrads[i] = (t_values[i] - self.oNodes[i])

		for i in range(self.nh):
			derivative = (1 - self.hNodes[i]) * (1 + self.hNodes[i]);
			sum_ = 0.0
			for j in range(self.no):
				sum_ += oGrads[j] * self.hoWeights[i, j]
			hGrads[i] = derivative * sum_

	def updateWeightsAndBiases(self, learnRate, hGrads, oGrads):
		for i in range(self.ni):
			for j in range(self.nh):
				self.ihWeights[i,j] += learnRate * hGrads[j] * self.iNodes[i]
		for i in range(self.nh):
			for j in range(self.no):
				self.hoWeights[i,j] += learnRate * oGrads[j] * self.hNodes[i];

		for i in range(self.nh):
			self.hBiases[i] += learnRate * hGrads[i] * 1.0 
		for i in range(self.no):
			self.oBiases[i] += learnRate * oGrads[i] * 1.0

	def crossEntropyError(self, x_values, t_values):
		sumError = 0.0
		for i in range(len(x_values)):
			y_values = self.computeOutputs(x_values[i])
			for j in range(self.no):
				if (y_values[j] * t_values[i][j] != 0):
					sumError += math.log(y_values[j] * t_values[i][j])
		return -1.0 * sumError / len(x_values)

	def accuracy(self, x_values, t_values):
		num_correct = 0
		num_wrong = 0
		for i in range(len(x_values)):
			y_values = self.computeOutputs(x_values[i])
			max_index = np.argmax(y_values)
			if abs(t_values[i][max_index] - 1.0) < 1.0e-5:
				num_correct += 1
			else:
				num_wrong += 1
		return (num_correct * 1.0) / (num_correct + num_wrong)

	@staticmethod
	def hypertan(x):
		if x < -20.0:
		    return -1.0
		elif x > 20.0:
		    return 1.0
		else:
		    return math.tanh(x)

	@staticmethod	  
	def softmax(oSums):
		result = np.zeros(shape=[len(oSums)], dtype=np.float32)
		max_ = max(oSums)
		divisor = 0.0
		for elem in oSums:
			divisor += math.exp(elem - max_)
		for i,elem in enumerate(oSums):
			result[i] =  math.exp(elem - max_) / divisor
		return result

