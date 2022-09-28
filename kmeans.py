import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import norm
from sklearn.datasets import make_circles
class KMeans:
    def __init__(self, data, k):
        #taking a copy of the dataset inside the object
        self.Dcopy = data
        #how many different clusters must be classified
        self.K = k
    #function to initialise initial centroids
    def init_centroids(self):
        #centroids are assigned
        self.centroids = self.Dcopy[0:self.K]
        #initialising a clusters array 
        self.clusters = []
    #function to calculate euclidean distance
    def eu_dist(self,x,y):
        #to know length for finding sum of squares
        ln=len(x)
        ls=0
        for z in range(ln):
            ls+=(x[z]-y[z])**2
            print(ls)
        return (ls)**(0.5)
    #main function that runs the algorithm
    def fit(self):
        #epsilon value for stopping the iteration 
        eps = 0.1
        #variable that holds consecutive value's difference
        diff = 1000000
        while (diff > eps):
            temp0 = []
            ln = len(self.Dcopy)
            #Loop to calculate the distance from centroids and assign clusters
            for x in range(ln):
                temp1 = []
                for y in range(self.K):
                    #finding distance of particular point with respect to initialised centroids
                    temp1.append(self.eu_dist(self.centroids[y],self.Dcopy[x]))
                #finding the minimum of the centroid distances to assign the point to a new cluster
                temp0.append(temp1.index(min(temp1)))
            #assigning cluster values
            self.clusters = np.array(temp0)
            print("centroids",self.centroids)
            print("clusters :",self.clusters)
            temp2 = np.array(self.centroids[0])
            print("temp2",temp2)
            # using the assigned cluster values to find mean and assign it to centroids
            for x in range(self.K):
                # finding values belonging to one cluster == 0,1,2
                ls = np.where(self.clusters == x)
                print("cluster array :",ls)
                print(x)
                ls1 = []
                for y in ls:
                    ls1.append(self.Dcopy[y])
                #calculate new centroids
                self.centroids[x] = np.mean(np.array(ls1), 1)
            diff = self.eu_dist(self.centroids[0],temp2)
            print("diff : ",diff)
            print("centroid value",self.centroids)
            print("temp2",temp2)


'''X=pd.read_csv("iris.csv")
x=np.array(X)
print(x)
for y in x:
    if (y[4]=='Iris-setosa'):
        y[4]=0
    elif (y[4]=='Iris-versicolor'):
        y[4]=1
    else:
        y[4]=2
print(x)''''''
variables,target=make_circles(1500)
obj = KMeans(variables,2)
obj.init_centroids()
obj.fit()


plt.scatter(variables[:,0],variables[:,1],c=obj.clusters)
plt.show()'''
r1 = 5;
r2=10;

dtheta = 1.5*(np.pi/180)
theta = np.linspace(0,2*np.pi,int(np.floor((2*np.pi)/dtheta)))
x1 = r1*np.cos(theta)
y1 = r1*np.sin(theta)
x1 = np.array(x1 + np.random.rand(*x1.shape))
y1 =  np.array(y1+np.random.rand(*y1.shape))

x2 = r2*np.cos(theta)
y2= r2*np.sin(theta)
x2 = np.array(x2  +np.random.rand(*x2.shape))
y2 = np.array(y2 + np.random.rand(*y2.shape))
#print(x1,x2,type(x1))
X=np.concatenate((x1,x2))
Y=np.concatenate((y1,y2))
plt.plot(x1,y1,'.')
plt.plot(x2,y2,'.')
vari=np.zeros([len(X),2])
for x in range(len(X)):
    vari[x]=[X[x],Y[x]]
print(X,Y,vari)
obj = KMeans(vari,2)
obj.init_centroids()
obj.fit()
plt.scatter(X,Y,c=obj.clusters)
plt.show()