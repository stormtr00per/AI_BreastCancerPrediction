import numpy as np

#Name:	Sarth Desai
#NetID:	sjd445

#---------------------------------------------------------------    KNN    -------------------------------------------------------------------------------
class KNN:
	def __init__(self, k):
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.X_train = X
		self.y_train = y

	def majorityVoting(self, index_row):			#Calculating distance array to get neighbours and returning the classification by votes
		new_arr = []
		distanceArr = [(self.distance(index_row, feature), feature) for feature in self.X_train]
		arr = ([feature[0] for feature in distanceArr])
		arr.sort()
		for index in range(self.k):
			new_arr.append(arr[index])
		most_votes = [vector[1] for vector in zip(distanceArr, self.y_train) if vector[0][0] in new_arr]
		return np.array(most_votes)

	def predict(self, X):
		final = np.zeros(len(X))
		for index, row in enumerate(X):
			final[index] = np.bincount(self.majorityVoting(row)).argmax()
		return np.array(final)


#------------------------------------------------------------     ID3     --------------------------------------------------------------------------------
class ID3:
	def __init__(self, nbins, data_range):
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size * norm_data).astype(int)
		return categorical_data

	def calculateEntropy(self, col):
		ele, count = np.unique(col, return_counts=True)
		entropy = np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count)) for i in range(len(ele))])
		return entropy

	def InfoGain(self, X):
		IG = []
		entropy = self.calculateEntropy(X)
		vals, count = np.unique(X, return_counts=True)
		weight = np.sum(
			[(count[i] / np.sum(count)) * self.calculateEntropy(X.where(X[i] == vals[i]).dropna()) for i in range(len(vals))])
		IG = entropy - weight
		return IG

	def train(self, X, y):
		categorical_data = self.preprocess(X)

	def predict(self, X):
		categorical_data = self.preprocess(X)
		return None


#-----------------------------------------------------------   Perceptron   ------------------------------------------------------------------------------
class Perceptron:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		for i in range(steps):
			index = np.random.randint(0, steps//100-1)
			index = index%y.size
			self.w = self.w + self.lr*(y[index]-self.activationFunction(X[index]))*X[index]
			self.b = self.b + self.lr*(y[index]-self.activationFunction(X[index]))
		return self

	def activationFunction(self, X):					#This is perceptron's activation function
		if sum(X*self.w) + self.b > 0:					#Source Chp:8 Slide:5
			return 1
		else:
			return 0

	def predict(self, X):
		S = np.zeros(len(X))
		for index, element in enumerate(X):
			S[index] = self.activationFunction(element)
		return np.array(S)


#-------------------------------------------------------------    MLP    ---------------------------------------------------------------------------------
class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:							#Source Chp:8 Slide:40
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def forward(self, input):
		self.i = input
		return np.dot(input, self.w) + self.b

	def backward(self, gradients):
		w_update = self.i.T.dot(gradients)
		x_update = gradients.dot(self.w.T)
		self.w = self.w - self.lr*w_update
		self.b = self.b - self.lr*gradients
		return x_update

class Sigmoid:							#Source Chp:8 Slide:25, 38
	def __init__(self):
		None

	def forward(self, input):
		self.i = input
		self.sigmoid_x = 1/(1 + np.exp(-self.i))
		return self.sigmoid_x

	def backward(self, gradients):
		sigmoidDerivative = (1-self.sigmoid_x)*self.sigmoid_x
		return gradients*sigmoidDerivative