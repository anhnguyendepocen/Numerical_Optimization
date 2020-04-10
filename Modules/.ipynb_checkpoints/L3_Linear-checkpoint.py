import numpy as np

class LogisticRegression:
    def __init__(self,step=0.1, iterations=501, verbose=False):
        self.step = step
        self.iterations = iterations
        self.verbose = verbose
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        #print('Theta shape: {}, Z shape: {} \n'.format(np.shape(theta),np.shape(X@theta)))
    
        for i in range(self.iterations):        
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -=  self.step * gradient
            
            if(self.verbose ==True and i % 250 == 0):
                print("Iteration "+str(i)+"\n Gradient: {}, Loss: {}, Theta: {}\n".format(
                    np.round(gradient,2), round(self.__loss(h,y),3), np.round(self.theta,2)))

    def predict(self, X):
        probability = self.__sigmoid(np.dot(X, self.theta))
        predict = (probability>0.5)*1
        return probability, predict
