import numpy as np 
from scipy.linalg import hadamard
from scipy.spatial import distance
import matplotlib.pyplot as plt


def makeHadamard(n):
	return hadamard(n, dtype=float)

def generate_x(n):
    return np.random.randint(2, size = (n))

def mechanism(my_x, n, H, sigma):
    Y = np.random.normal(loc =0, scale =sigma, size=(n))
    x = my_x
    a = ((1/n)*(H)@(x) + Y)
    return a

def reconstruct(x, n, H, sigma):
    a = mechanism(x, n,H,sigma)
    xhat = H @ a
    xhat[np.absolute(xhat) < 0.5] = 0
    xhat[np.absolute(xhat) >= 0.5] = 1
    return xhat

def hamming(x,n,H,sigma):
    return distance.hamming(x,reconstruct(x,n,H,sigma))

def experiment(n,sigma):
    hams = []
    my_sig = str(sigma)
    H = makeHadamard(n)
    for i in range(20):
        x = generate_x(n)
        hams = hams + [hamming(x,n,H,sigma)]
    print("sigma = ",my_sig.ljust(13),end = "")
    print("=\t\t%f\t %f" % (np.average(hams),np.std(hams)))
    return (np.average(hams),np.std(hams))


def plotting():
    print("\n\n")
    ns = [128,512,2048,8192]
    sigmas = [1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256]
    for n in ns:
        a=[]
        print("\t\t\t\t   Avg\t\t Std Dev")
        print("n=",n)
        for s in sigmas:
            a = a + [experiment(n,s)]
        e = [i[1] for i in a]
        plt.errorbar([1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256],[i[0] for i in a],yerr=e,fmt='-o',label="n="+str(n))
        plt.axis([-0.03,0.55,-0.05,0.55])
        plt.xlabel("sigma")
        plt.ylabel("mean fraction of missed bits")
        plt.title("Mean fraction of missed bits vs. Sigma,n")
        plt.savefig("1a.png")
        plt.legend(loc="lower right")
        print("\n")
    plt.show()



plotting()









