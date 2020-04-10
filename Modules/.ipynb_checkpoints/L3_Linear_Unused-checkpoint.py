from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class LinearClass(object):
    def __init__(self, X, y, graph=False):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=1)
         
    def knn(self):
        fit = KNeighborsClassifier().fit(self.X, self.y)      
        y_pred = fit.predict(self.X_test)
        print(confusion_matrix(self.y_test,y_pred))
        print(classification_report(self.y_test, y_pred))
        graph:
            plt.scatter(x,y)
        
    def svm(self):
        fit = LinearSVC().fit(self.X, self.y)
        y_pred = fit.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_test, y_pred))   
        
    def logistic():
        fit = SVC().fit(self.X, self.y)
        y_pred = fit.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_test, y_pred))   
