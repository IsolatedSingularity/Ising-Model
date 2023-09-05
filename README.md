# Ising-Model
###### For the graduate statistical mechanics course taught by [Professor Martin Grant](http://www.physics.mcgill.ca/~grant/) at [McGill University](https://www.mcgill.ca/).

![alt text](https://github.com/IsolatedSingularity/Ising-Model/blob/main/Plots/CaptureII.PNG)

## Objective

Computing a 2D Ising model simulation with Glauber &amp; Metropolis rules.

## Code Functionality

It begins by importing necessary libraries such as NumPy for numerical operations, Matplotlib for plotting, and random for random number generation. It also imports curve_fit from SciPy for curve fitting. It then sets various parameters for the simulation, including the size of the 2D lattice (size), the number of time steps (timeSteps), the number of runs (runSteps), and the critical temperature (critTemp) for a phase transition. It defines a function deltaEnergy to calculate the change in energy when a spin is flipped in a local 2D neighborhood. This function considers the neighboring spins and computes the energy difference due to flipping. It defines two fitting functions, magnetizationFunction and tauFunction. These functions are used for curve fitting to extract specific parameters from the simulation results. The code performs a Monte Carlo simulation by iterating through different temperature values (defined in temperatures). For each temperature, it iterates through multiple runs (runSteps) and simulates the evolution of spins over time steps. In each time step, spins are randomly chosen and flipped based on the Glauber condition, which depends on temperature and energy differences. This simulates the thermal fluctuations and interactions between spins. The code calculates the magnetization per spin at each time step and averages it over multiple runs to smoothen the magnetization curve. It fits the magnetization curve to a specific function (magnetizationFunction) to extract the relaxation time (τ). This process is repeated for different temperatures, resulting in an array of τ values (tauArray) for different temperatures.
