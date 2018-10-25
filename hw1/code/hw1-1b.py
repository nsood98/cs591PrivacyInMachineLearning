import numpy as np 
from scipy.spatial import distance
import matplotlib.pyplot as plt



def makeB(m,n):
	return np.random.randint(2, size = (m,n))

def generate_x(n):
    return np.random.randint(2, size = (n,1))

def mechanism(my_x, m,n, B, sigma):
    Y = np.random.normal(loc =0, scale =sigma, size=(m,1))
    x = my_x
    a = ((1/n)*((B)@(x)) + Y)
    return a

def reconstruct(x, m,n, B, sigma):
    a = mechanism(x,m,n,B,sigma)
    xhat = np.linalg.lstsq((B/n),a,rcond=-1)[0]
    xhat[np.absolute(xhat) < 0.5] = 0
    xhat[np.absolute(xhat) >= 0.5] = 1
    xhat = [xi[0] for xi in xhat]
    return xhat

def hamming(x,m,n,B,sigma):
    return distance.hamming((x).flatten(),reconstruct(x,m,n,B,sigma))


def experiment(m,n,sigma):
    hams = []
    my_sig = str(sigma)
    for i in range(20):
        B = makeB(m,n)
        x = generate_x(n)
        hams = hams + [hamming(x,m,n,B,sigma)]
    print("sigma = ",my_sig.ljust(13),end = "")
    print("=\t\t%f\t %f" % (np.average(hams),np.std(hams)))
    return (np.average(hams),np.std(hams))


def plotting():
    print("\n\n")
    ns = [128,512,2048,8192]
    sigmas = [1/4,1/8,1/16,1/32,1/64,1/128,1/256]
    for n in ns:
        ms = [int(1.1*n),4*n,16*n]
        for m in ms:
            a=[]
            print("\t\t\t\t   Avg\t\t Std Dev")
            print("m=",m,"n=",n)
            for s in sigmas:
                a = a + [experiment(m,n,s)]
            e = [i[1] for i in a]
            plt.errorbar([1/4,1/8,1/16,1/32,1/64,1/128,1/256],[i[0] for i in a],yerr=e,fmt='-o',label="m="+str(m))
            plt.axis([-0.03,0.3,-0.05,0.6])
            plt.xlabel("sigma")
            plt.ylabel("mean fraction of missed bits")
            plt.title("Mean fraction of missed bits vs. Sigma,m,n="+str(n))
            plt.legend(loc="lower right")
            print("\n")
        plt.savefig("1b-n"+str(n)+".png")
        plt.show()


plotting()









