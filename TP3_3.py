import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class POO_model:
    def __init__(self):
        self.W = None
        self.b = None
    
    def initialisation(self,X):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
    
    def model(self,X):
        Z = X.dot(self.W) + self.b
        # print(Z.min())
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def log_loss(self,A, y):
        epsilon = 1e-15
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def gradients(self,A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)
    
    def update(self,dW, db, learning_rate):
        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db
        

    def predict(self,X):
        A = self.model(X)
        # print(A)
        return A >= 0.5 
    def artificial_neuron(self, X_train, y_train, X_test, y_test, learning_rate, n_iter):
        # initialisation W, b
        self.initialisation(X_train)

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for i in tqdm(range(n_iter)):
            A = self.model(X_train)

            if i %10 == 0:
                # Train
                train_loss.append(self.log_loss(A, y_train))
                y_pred = self.predict(X_train)
                train_acc.append(accuracy_score(y_train, y_pred))

                # Test
                A_test = self.model(X_test)
                test_loss.append(self.log_loss(A_test, y_test))
                y_pred = self.predict(X_test)
                test_acc.append(accuracy_score(y_test, y_pred))

            # mise a jour
            dW, db = self.gradients(A, X_train, y_train)
            self.update(dW, db, learning_rate)


        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='train acc')
        plt.plot(test_acc, label='test acc')
        plt.legend()
        plt.show()

