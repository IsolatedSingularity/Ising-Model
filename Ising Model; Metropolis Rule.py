#%% Retrieving Libraries
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.optimize import curve_fit

# %% Defining System Parameters
size = 50 #size x size 2D lattice
timeSteps = 150 #amount of time steps / repeated Monte Carlo steps
runSteps = 40 #amount of runs / repeated time steps
critTemp = 2.269 #critical temperature in units J/k

# %% Defining change of energy of a spin flip in a local 2D Neighbourhood (:= Nier)
def deltaEnergy(firstSpin,secondSpin): #firstSpin/secondSpin: spin at the indexed position in lattice
    tempRandomNumber = rd.random()

    if firstSpin == 0:
        topNier = spinLattice[size-1][secondSpin]
    else:
        topNier = spinLattice[firstSpin-1][secondSpin]

    if firstSpin == size-1:
        bottomNier = spinLattice[0][secondSpin]
    else:
        bottomNier = spinLattice[firstSpin+1][secondSpin]

    if secondSpin == 0:
        leftNier = spinLattice[firstSpin][size-1]
    else:
        leftNier = spinLattice[firstSpin][secondSpin-1]

    if secondSpin == size-1:
        rightNier = spinLattice[firstSpin][0]
    else:
        rightNier = spinLattice[firstSpin][secondSpin+1]

    finalEnergyDifference = 2*spinLattice[firstSpin][secondSpin]*(topNier+bottomNier+leftNier+rightNier) #considering contributions of Niers when flipping spin

    return finalEnergyDifference

#%% Defining fitting functions
def magnetizationFunction(t, τ, amp): #used for extracting values of τ
    return amp*np.exp(t/(-τ))

def tauFunction(T,A,μ): #used for extracting values of A and μ
    return A * (T-critTemp)**(-μ)

# %% Performing Monte Carlo spin flips & extracting desired parameters
temperatures = np.arange(2.27,3,0.1) #considering a range of temperatures near critTemp to fit for A and μ in τ(T)
tauArray = np.zeros(len(temperatures)) #initializing null valued array for values of τ

for tempSteps in range(0,len(temperatures)): #iterating through different temperature values

    magnetizationArray = np.zeros(timeSteps) #empty array of magentizations per spin

    for step in range(0,runSteps): #iterating through multiple runs and averaging over them to smoothen m(t) curve

        spinLattice =  np.full((size,size),1) #initial lattice with all spin up, could have been alternatively all spin down
        tempMagnetizationArray = np.zeros(timeSteps)

        for time in range(0,timeSteps): #iterating through times 0 to timeSteps

            for iteration in range(0,(size**2)-1): #one Monte Carlo step for flipping spins in lattice
                randomNumber1 = rd.random()
                randomNumber2 = rd.random()
                randomNumber3 = rd.random()
                i = int(randomNumber1*size)
                j = int(randomNumber2*size)
                tempEnergyDifference = deltaEnergy(i,j)

                if tempEnergyDifference <= 0: #Metropolis condition for flipping spin: deltaE <= 0, thus W=1 (W is the transition rate)
                    spinLattice[i][j] = -spinLattice[i][j]
                else:
                    if randomNumber3 < np.exp(-tempEnergyDifference/(temperatures[tempSteps])): #deltaE>0, thus W=exp(-deltaE/kT)
                        spinLattice[i][j] = -spinLattice[i][j]
            
            magnetizationPerSpin = (1/(size**2))*(spinLattice.sum()) #magnetization per spin at time t
            tempMagnetizationArray[time] += magnetizationPerSpin #adding magnetization array for current run to empty array, this is to average over runs once divided by amount of runs

        magnetizationArray += tempMagnetizationArray
        normalizedMagnetizationArray = magnetizationArray / runSteps #averaging over runs for every time step

    
    xValues = np.linspace(0,timeSteps,timeSteps) #defining horizontal range to perform curve fit
    poptMag, pcovMag = curve_fit(magnetizationFunction, xValues, normalizedMagnetizationArray) #fitting for τ from m(t)
    tauArray[tempSteps] += poptMag[0]


# %% Plotting m vs t
plt.plot(xValues, normalizedMagnetizationArray)

# %% Fitting for A and μ from τ(T)
plt.plot(temperatures,tauArray)
poptTau, pcovTau = curve_fit(tauFunction, temperatures, tauArray)

# %% Results 

#params: 
# size = 50 #size x size 2D lattice
# timeSteps = 150
# runSteps = 40
# critTemp = 2.269

# temperatures = np.arange(2.27,3,0.1)

#A, mu = [50.19911104  0.39570381]