import numpy as np
import csv

def loadCSV(filename):
    '''
    function to load dataset
    '''
    with open(filename,"r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
    #If it shows an error while training convert float to int
    return np.array(dataset)

def sigmoid(x):
    x = np.float64(x)
    return ( 1.0 / (1.0 + np.exp(-x)))


def cost_function (X, Y, theta, lmbda):
    h = sigmoid(X @ (theta));
    m = len(X)
    #print("H: ",h)
    J = (-1/m)*np.sum(Y*np.log(h) + (1-Y)*np.log(1-h)) + (lmbda/(2*m))*np.sum(theta[1:] * theta[1:])
    print("J : ",J)
    #grad = np.zeros((1,len(theta)))
    #print(h-Y)
    grad = (1/m)*np.dot(np.transpose(X),(h-Y))
    #grad[1:] = (1/m)*np.sum((h-Y)[:,np.newaxis]*X[:,1:]- (lmbda/m)*theta[1:]
    #grad[1:] = grad[1:] - (lmbda/m)*theta[1:]
    #print(grad)
    return [J,grad]

def gradient_descent (X,Y,theta,lr,conv,lmbda):
    [current,grad] = cost_function(X,Y,theta,lmbda)
    prev = 0.0;
    m = len(X)
    h = sigmoid(X @ np.transpose(theta))
    while(abs(current - prev) >= conv):
        theta = theta - (lr*grad)
        prev = current;
        [current,grad] = cost_function(X,Y,theta,lmbda)
        print(abs(current - prev) )
        print(conv)
    return theta

def one_vs_all(X,Y, Num_labels):
    [m, n] = np.shape(X)
    all_theta = np.zeros((Num_labels,n), dtype = np.float64)
    lmbda = 0
    lr = 0.000001
    conv = 0.000001
    print("Training\n")
    for i in range(Num_labels):
        Y_one = (Y==(i))*1
        all_theta[i,:] = gradient_descent(X,Y_one,all_theta[i,:],lr, conv,lmbda)
        print("Trained ",i)
    return all_theta

def predict(X,Y,all_theta, Num_labels):

    h = sigmoid(X @ np.transpose(all_theta))
    prediction = h.argmax(axis=1)
    return prediction

def accuracy(pred,Y):
    err = (pred == Y)
    err = err*1
    acc = (np.sum(err)/len(err))*100
    print("Accuracy is: ",acc,"\n")
    return acc

train = loadCSV("mnist_train.csv")
test = loadCSV("mnist_test.csv")

N_train = len(train)
N_test = len(test)
X_train = train[:,1:]
X_test = test[:,1:]
Y_train = train[:,0]
Y_test = test[:,0];

print(N_train)
print(N_test)
o_train = np.ones((N_train,1),dtype= np.float64)
o_test = np.ones((N_test,1), dtype= np.float64)
X_train = np.concatenate((o_train,X_train),1)
X_test = np.concatenate((o_test,X_test),1)

Num_labels = 10;
Num_features = len(train[0]) + 1

#Uncomment these lines to train.
all_theta = loadCSV("theta.csv");
#all_theta = one_vs_all(X_train,Y_train,Num_labels)
#np.savetxt("theta.csv",all_theta,delimiter=',')
pred1 = predict(X_train, Y_train, all_theta,Num_labels)
acc1 = accuracy(pred1,Y_train)

print("Testing\n")
pred2 = predict(X_test, Y_test, all_theta,Num_labels)
acc2 = accuracy(pred2,Y_test)
