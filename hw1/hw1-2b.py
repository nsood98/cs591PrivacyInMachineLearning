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


#Part b) Input to attacker is vector a (noisy counters) and vector w (guesses)

def generate_w(x):
    w = np.array(x)
    for i in range(len(w)):
        if (np.random.randint(3) == 2):
            if w[i] == 0:
                w[i] = 1
            else:
                w[i] = 0
    return w

#denoising algorithm
def remove_noise(a,w):
    if a[0] == 2:
        a[0] = 1
    elif a[0] == 1:
        a[0] = w[0]             #improves chance of correctness of 1st bit from 1/2 to 2/3
    for i in range(len(a)-1):
        if a[i+1]- a[i] > 1:
            a[i+1] = a[i+1]-1
        elif a[i]-a[i+1] > 1:
            a[i] = a[i]-1 
    return a

#reconstructs x-hat from denoised vector a
def reconstruct(a,w):
    x = []
    if a[0] == 0:
        x = x + [0]
    else:
        x = x + [1]
    a = remove_noise(a,w)
    for i in range(len(a)-1):
        if a[i+1] > a[i]:
            if w[i+1] == 1:         #checks if the guesses had a 1 in that index
                x = x + [1]         #made it worse by 3-4% worse with only this implemented
            else:
                if (np.random.randint(2) == 1):  #with this, makes it worse by about 1-2%
                    x = x + [0]
                else:
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
        w = generate_w(ex[0])
        hams = hams + [distance.hamming(ex[0],reconstruct(ex[2],w))]
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
    plt.errorbar([100,500,1000,5000],[i[0] for i in a],yerr=e,fmt='-o',label="2b")
    plt.axis([0,6000,-0.05,0.55])
    plt.xlabel("sigma")
    plt.ylabel("mean fraction of missed bits")
    plt.title("Mean fraction of missed bits vs. n,w")
    plt.savefig("2b.png")
    plt.show()



plotting()



