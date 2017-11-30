import math

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

		self.rnd = random.Random(10) # allows multiple instances
		self.initializeWeights()
	
	def initializeWeights(self):
		lo = -0.05 
		hi = 0.05
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

	def computeOutputs(self, xValues):
		hSums = np.zeros(shape=[self.nh], dtype=np.float32)
		oSums = np.zeros(shape=[self.no], dtype=np.float32)

		for i in range(self.ni):
			self.iNodes[i] = xValues[i]

		for j in range(self.nh):
			for i in range(self.ni):
				hSums[j] += self.iNodes[i] * self.ihWeights[i,j]

		for j in range(self.nh):
			hSums[j] += self.hBiases[j]
		  
		for j in range(self.nh):
			self.hNodes[j] = self.hypertan(hSums[j])

		for k in range(self.no):
			for j in range(self.nh):
				oSums[k] += self.hNodes[j] * self.hoWeights[j,k]

		for k in range(self.no):
			oSums[k] += self.oBiases[k]

		softOut = self.softmax(oSums)
		for k in range(self.no):
			self.oNodes[k] = softOut[k]
		  
		result = np.zeros(shape=self.no, dtype=np.float32)
		for k in range(self.no):
			result[k] = self.oNodes[k]
		  
		return result

	def MeanCrossEntropyError(self, trainData):  # on train or test data matrix
		sumSquaredError = 0.0
		x_values = np.zeros(shape=[self.ni], dtype=np.float32)
		t_values = np.zeros(shape=[self.no], dtype=np.float32)

		for i in range(len(trainData)):  # walk thru each data item
			for j in range(self.ni):  # peel off input values from curr data row 
				x_values[j] = trainData[i, j]
		
			for j in range(self.no):  # peel off tareget values from curr data row
				t_values[j] = trainData[i, j+self.ni]

			y_values = self.computeOutputs(x_values)  # computed output values
		  
			for j in range(self.no):
				sumError += log(y_values[j] * t_values[j])
			
		return -1.0 * sumError / len(trainData)

	def accuracy(self, tdata):  # train or test data matrix
	    num_correct = 0; num_wrong = 0
	    x_values = np.zeros(shape=[self.ni], dtype=np.float32)
	    t_values = np.zeros(shape=[self.no], dtype=np.float32)

	    for i in range(len(tdata)):  # walk thru each data item
	      for j in range(self.ni):  # peel off input values from curr data row 
	        x_values[j] = tdata[i,j]
	      for j in range(self.no):  # peel off tareget values from curr data row
	        t_values[j] = tdata[i, j+self.ni]

	      y_values = self.computeOutputs(x_values)  # computed output values)
	      max_index = np.argmax(y_values)  # index of largest output value 

	      if abs(t_values[max_index] - 1.0) < 1.0e-5:
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
		    return math.Tanh(x)

	@staticmethod	  
	def softmax(oSums):
		result = np.zeros(shape=[len(oSums)], dtype=np.float32)
		max_ = max(oSums)
		divisor = 0.0
		for elem in range(oSums):
		   divisor += math.exp(elem - max_)
		for elem,i in enumerate(oSums):
		  result[i] =  math.exp(elem - max_) / divisor
		return result