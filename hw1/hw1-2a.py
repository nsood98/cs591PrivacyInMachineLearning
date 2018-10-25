import numpy as np 
from scipy.spatial import distance
import matplotlib.pyplot as plt


#makes sequence of x bits, each x(i) s.t. {0,1}
#Taken from Piazza, posted by Adam Smith
def generate_data(n):
    x = np.random.randint(2, size = n)
    exact_answers = np.cumsum(x) # This computes the array of prefix hamss of x.
    #exact_answers is the total number of people who clicked on the ad out of n people
    z = np.random.randint(2, size = n)
    a = exact_answers + z         # a = noisy answers
    return (x, exact_answers, a)


#Part a) Input to attacker is vector a (noisy counters)

#denoising algorithm
def remove_noise(a):
    if a[0] == 2:
        a[0] = 1
    elif a[0] == 1:
        a[0] = np.random.randint(2)
    for i in range(len(a)-1):
        if a[i+1]- a[i] > 1:
            a[i+1] = a[i+1]-1
        elif a[i]-a[i+1] > 1:
            a[i] = a[i]-1 
    return a

#reconstructs x-hat from denoised vector a
def reconstruct(a):
    x = []
    if a[0] == 0:
        x = x + [0]
    else:
        x = x + [1]
    a = remove_noise(a)
    for i in range(len(a)-1):
        if a[i+1] > a[i]:
            x = x + [1]
        else:
            x = x + [0]
    x = np.array(x)
    return x

#runs the 20 trials, calculates mean and std dev
def experiment(n):
    hams = []
    for i in range(20):
        ex = generate_data(n)
        hams = hams + [distance.hamming(ex[0],reconstruct(ex[2]))]
    print("\t\t%f\t %f" % (np.average(hams),np.std(hams)))
    return np.average(hams),np.std(hams)

def plotting():
    print("\n\n")
    ns = [100,500,1000,5000]
    a=[]
    e = []
    print("\t\t   Avg\t\t Std Dev")
    for n in ns:
        print("n=",n,end="")
        a = a + [experiment(n)]
        e = [i[1] for i in a]
    print("\n\n")
    plt.errorbar([100,500,1000,5000],[i[0] for i in a],yerr=e,fmt='-o',label="2a")
    plt.axis([0,6000,-0.05,0.55])
    plt.xlabel("sigma")
    plt.ylabel("mean fraction of missed bits")
    plt.title("Mean fraction of missed bits vs. n")
    plt.savefig("2a.png")
    plt.show()



plotting()



