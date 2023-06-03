#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:58:49 2023

@author: adamthepig
"""

#------------------------------- MODULE IMPORTS -------------------------------

import sys
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
#--------------------------------- CONSTANTS ----------------------------------

#- Simulation Constants
#-----------------------
#- Number of sets of simulations run
SET_NUM = 10
#- Number of simulations run per set
SIMULATION_NUM = 10
#- Number of time steps in the simulation
TIME_STEPS = 3000
#- Can agents move diagonally? (affects distance calc)
DIAGONAL_MOVEMENT = True
#- Do agents reproduce asexually?
ASEXUAL = True

#- Display Constants
#--------------------
#- Grid size
WIDTH = 150 #- Horizontal
HEIGHT = 150 #- Vertical
#- The number of pixels for each grid location
PIXEL_SIZE = 5
#- FPS of visualization
FPS = 30
#- The visualization colors
BACKGROUND_COLOR = (255, 255, 255)
PREY_COLOR = (0, 0, 255)
PREDATOR_COLOR = (255, 0, 0)
PREDATOR_STUN_COLOR = (100, 100, 100)
#- Plant colors are interpolated between these two
PLANT_GROWN_COLOR = (0, 255, 0)
PLANT_UNGROWN_COLOR = (255, 255, 0)

#- Predator Constants
#---------------------
#- Starting Number of Agents
PREDATOR_START_NUM = 110
#- Starting Energy Value
PREDATOR_START_ENERGY = 50
#- Energy Gain by Eating
PREDATOR_EAT_ENERGY = 50
#- Maximum Energy Value
PREDATOR_MAX_ENERGY = 100
#- Reproduction Timer (How many cycles between possible reproductions)
PREDATOR_REPRODUCTION_TIME = (15, 25)
#- Eyesight Radius
PREDATOR_EYESIGHT = 10
#- Energy Spent in Single Move Cycle
PREDATOR_MOVE_ENERGY = 1.1
#- Number of Cycles stunned
STUN_TIME = 5
#- Radius that another predator has to be in for reproduction to occure
PREDATOR_REPRODUCTION_RANGE = 5
#- Energy threshold for reproduction
PREDATOR_REPRODUCTION_THRESHOLD = 80

#- Prey Constants
#-----------------
#- Starting Number of Agents
PREY_START_NUM = 200
PREY_SPEED = 0.85
#- Starting Energy Value
PREY_START_ENERGY = 50
#- Energy Gain by Eating 
PREY_EAT_ENERGY = 100
#- Maximum Energy Value
PREY_MAX_ENERGY = 100
#- Reproduction Timer (How many cycles between possible reproductions)
PREY_REPRODUCTION_TIME = (15, 25)
#- Eyesight Radius
PREY_EYESIGHT = 5
#- Energy Spent in Single Move Cycle
PREY_MOVE_ENERGY = 1
#- Energy Threshold for Eating
PREY_HUNGRY = 90
#- Percent chance of stunning predators
STUN_CHANCE = 0.5
#- Percent chance of killing predators
PREY_KILL_CHANCE = 0.0
#- Radius that another prey has to be in for reproduction to occure
PREY_REPRODUCTION_RANGE = 5
#- Energy threshold for reproduction
PREY_REPRODUCTION_THRESHOLD = 80

#- Plant Constants
#------------------
#- The time required for plants to regrow after being eaten
PLANT_REGROWTH_TIME = 250

#- Analysis Constants
#------------------
#- Whether each plot should be displayed after a simulation, set, or set of
#  sets.
LAG_PLOT_DISLPAYS = {"sets": True, "set": True, "simulation": True}
POPULATION_PLOT_DISPLAYS = {"sets": True, "set": True, "simulation": True}

def runSets():
    """
    Runs multiple sets of simulations, then prints average populations at the
    end of the simulation and the standard deviation of the populations.
    """
    preyPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    predatorPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    for i in range(SET_NUM):
        print(f'Running Set {i+1}', end='\r')
        preyPopulations[i], predatorPopulations[i] = runSet()
    print(
f"""
Populations at end of simulation
Averages
    Prey: {np.average(preyPopulations[:, :])}
    Predators: {np.average(predatorPopulations[:, :])}
Standard Deviations:
    Prey: {np.std(np.average(preyPopulations[:, :, -1], axis=0))}
    Predators: {np.std(np.average(predatorPopulations[:, :, -1], axis=0))}
"""
        )
    if POPULATION_PLOT_DISPLAYS["sets"]:
        plotPopulations(np.average(preyPopulations, axis=(0, 1)),
                        np.average(predatorPopulations, axis=(0, 1)))
    if LAG_PLOT_DISLPAYS["sets"]:
        plotLagCorrelation(np.average(preyPopulations, axis=(0, 1)),
                           np.average(predatorPopulations, axis=(0, 1)))

def runSet():
    """
    Runs a single set of simulations.

    Returns
    -------
    preyPopulations : 2d scalar array
        The number of prey alive at each time step in each simulation
    predatorPopulations : 2d scalar array
        The number of predators alive at each time step in each simulation
    """
    preyPopulations = np.zeros((SIMULATION_NUM, TIME_STEPS))
    predatorPopulations = np.zeros((SIMULATION_NUM, TIME_STEPS))
    i = 0
    while i < SIMULATION_NUM:
        startTime = time.time()
        preyPopulations[i], predatorPopulations[i] = runSimulation(False)
        if preyPopulations[i, -1] == 0 or predatorPopulations[i, -1] == 0:
            print(f'sim {i+1} failed, rerunning')
            continue
        print(f"Finished Simulation {i+1}, Time Taken: {round(time.time() - startTime, 3)} seconds")
        i += 1
    if POPULATION_PLOT_DISPLAYS["set"]:
        plotPopulations(np.average(preyPopulations, axis=0),
                        np.average(predatorPopulations, axis=0))
    if LAG_PLOT_DISLPAYS["set"]:
        plotLagCorrelation(np.average(preyPopulations, axis=0),
                           np.average(predatorPopulations, axis=0))
    return preyPopulations, predatorPopulations

def runSimulation(shouldVisualize = False):
    """
    Runs a single simulation.

    Parameters
    ----------
    shouldVisualize : bool
        Whether to visualize the simulation using pygame
    Returns
    -------
    preyPopulations : 1d scalar array
        The number of prey alive at each time step
    predatorPopulations : 1d scalar array
        The number of predators alive at each time step
    """
    t('X')
    preyPopulations = np.zeros(TIME_STEPS)
    predatorPopulations = np.zeros(TIME_STEPS)
    #- Initializes the program
    prey, preyMask, predators, predatorMask, plants = initialize()
    if shouldVisualize:
        screen = initVisualization()
    t('X')
    for i in range(TIME_STEPS):
        t('TIMESTEP')
        #print(f'Running...{i}', end='\r')
        #- stop simulation if there are no prey or no predators alive
        if np.any(preyMask) and np.any(predatorMask):
            if shouldVisualize:
                screen.fill((255,255,255))
                #- Visualizes the grid
                if not visualize(screen, preyMask, predators, predatorMask,
                                 plants):
                    break
            t('X')
            #- Runs a single feed cycle
            t('1.1')
            feed(prey, preyMask, predators, predatorMask, plants)
            t('1.2')
            movePredators(preyMask, predators, predatorMask)
            movePrey(prey, preyMask, predatorMask, plants)
            reproduce(prey, preyMask, predators, predatorMask)
            t('1.5')
            preyPopulations[i] = np.count_nonzero(preyMask)
            predatorPopulations[i] = np.count_nonzero(predatorMask)
            t('1.6')
        else:
            break
    if shouldVisualize:
        pygame.quit()
    if POPULATION_PLOT_DISPLAYS["simulation"]:
        plotPopulations(preyPopulations, predatorPopulations)
    if LAG_PLOT_DISLPAYS["simulation"]:
        plotLagCorrelation(preyPopulations, predatorPopulations)
    return (preyPopulations, predatorPopulations)

def initialize():
    """
    Creates arrays and spawns prey, predators, and plants. Ensures that prey
    and predators do not overlap.

    Returns
    -------
    prey : 3d scalar array
        First two dimension correspond to position, third dimension is energy
        and time until possible reproduction.
    preyMask : 2d boolean array
        The locations that contain prey
    predators : 3d scalar array
        First two dimension correspond to position, third dimension is energy,
        time until possible reproduction, and stun time
    predatorMask : 2d boolean array
        The locations that contain predators
    plants : 2d scalar array
        The time until plants will be regrown
     : 2d boolean array
        The locations that contain plants
    """
    #- Create arrays
    prey = np.zeros((WIDTH, HEIGHT, 2))
    preyMask = np.zeros((WIDTH, HEIGHT), dtype=bool)
    predators = np.zeros((WIDTH, HEIGHT, 3))
    predatorMask = np.zeros((WIDTH, HEIGHT), dtype=bool)
    """
    global PREY_START_NUM
    global PREDATOR_START_NUM
    PREY_START_NUM = 1
    PREDATOR_START_NUM = 2
    preyMask[10, 10] = True
    predatorMask[13, 10] = True
    predatorMask[13, 11] = True
    """
    #- Create masks
    i = 0
    while i < PREY_START_NUM:
        x = random.randint(1, HEIGHT - 2)
        y = random.randint(1, WIDTH - 2)
        #- Ensure that there is not already a prey there
        if not preyMask[y, x]:
            preyMask[y, x] = True
            i += 1
    i = 0
    while i < PREDATOR_START_NUM:
        x = random.randint(1, HEIGHT - 2)
        y = random.randint(1, WIDTH - 2)
        #- Ensure that there is not already a prey or predator there
        if not predatorMask[y, x] and not preyMask[y, x]:
            predatorMask[y, x] = True
            i += 1
    #- Sets prey and predator start values
    prey[preyMask, 0] = PREY_START_ENERGY
    prey[preyMask, 1] = np.random.randint(
        PREY_REPRODUCTION_TIME[0], PREY_REPRODUCTION_TIME[1], PREY_START_NUM)
    predators[predatorMask, 0] = PREDATOR_START_ENERGY
    predators[predatorMask, 1] = np.random.randint(
        PREDATOR_REPRODUCTION_TIME[0], PREDATOR_REPRODUCTION_TIME[1],
        PREDATOR_START_NUM)
    #- Initializes plants
    plants = np.zeros((WIDTH, HEIGHT))
    return (prey, preyMask, predators, predatorMask, \
            plants)

def plotPopulations(preyPopulations, predatorPopulations):
    arange = np.arange(preyPopulations.size)
    fig, ax = plt.subplots()
    ax.plot(arange, preyPopulations, label="Prey Populations")
    ax.plot(arange, predatorPopulations, label="Predator Populations")
    plt.title("Prey and Predator Populations")
    plt.xlabel("Time (timesteps)")
    plt.ylabel("Populations")
    plt.show()

def plotLagCorrelation(preyPopulations, predatorPopulations,
                       maxLag = -1):
    #- Can't use a parameter in the default value of another paramter, so must
    #  do this instead
    if maxLag == -1:
        maxLag = int(np.size(preyPopulations) / 2)
    plt.xcorr(preyPopulations, predatorPopulations, normed=True,\
              usevlines=True, maxlags = maxLag)
    plt.xlabel("Lag (timesteps)")
    plt.ylabel("Correlation")
    plt.title("Lag Cross-Correlation Plot Between\n\
preyPopulations[i + lag] and predatorPopulations[i]")
    plt.show()
    
def initVisualization():
    """
    Begins the visiualization.

    Returns
    -------
    screen : Surface
        The surface to draw stuff on
    """
    #- Begins the visualization
    pygame.init()
    screen = pygame.display.set_mode([WIDTH * PIXEL_SIZE, HEIGHT * PIXEL_SIZE])
    #- Sets background color
    screen.fill((255, 255, 255))
    return screen

visualizing = True
def visualize(screen, preyMask, predators, predatorMask, plants):
    """
    Visualizes the prey, predators, and plants using the constant colors and
    constant background color. Also uses PIXEL_SIZE and the grid size given by
    WIDTH and HEIGHT.

    Parameters
    ----------
    screen: Surface
        The surface to draw stuff on
    preyMask : 2d boolean array
        The locations that contain prey
    predatorMask : 2d boolean array
        The locations that contain predators
    plants : 2d scalar array
        The time until plants will be regrown
     : 2d boolean array
        The locations that contain plants
    """
    clock = pygame.time.Clock()
    global visualizing
    #- Checks to see if the window has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                visualizing = not visualizing
    if visualizing:
        #- Iterates through grid
        for i in range(HEIGHT):
            for j in range(WIDTH):
                #- Draws a rectangle for each prey, predator, and plant
                rect = pygame.Rect(j * PIXEL_SIZE, i * PIXEL_SIZE,
                                   PIXEL_SIZE, PIXEL_SIZE)
                if preyMask[i, j]:
                    pygame.draw.rect(screen, PREY_COLOR, rect)
                elif predatorMask[i, j]:
                    if predators[i,j, 2] > 0:
                        pygame.draw.rect(screen, PREDATOR_STUN_COLOR, rect)
                    else:    
                        pygame.draw.rect(screen, PREDATOR_COLOR, rect)
                else:
                    plantUngrownColor = np.array(PLANT_UNGROWN_COLOR)
                    plantGrownColor = np.array(PLANT_GROWN_COLOR)
                    #- Interpolates color between plantUngrownColor and
                    #  plantGrownColor depending on how grown the plant is
                    plantColor = plantUngrownColor * plants[i, j] / \
                        PLANT_REGROWTH_TIME + plantGrownColor * (1 -
                        (plants[i, j] / PLANT_REGROWTH_TIME))
                    pygame.draw.rect(screen, plantColor, rect)
        #- Updates display
        pygame.display.flip()
        clock.tick(FPS)
    return True

def feed(prey, preyMask, predators, predatorMask, plants):
    """
    Makes predators eat prey and prey eat plants. Prey also stun predators and
    kills them depending on constant chances. Finally, regrows plants.

    Parameters
    -------
    prey : 3d scalar array
        First two dimension correspond to position, third dimension is energy
        and time until possible reproduction.
    preyMask : 2d boolean array
        The locations that contain prey
    predators : 3d scalar array
        First two dimension correspond to position, third dimension is energy,
        time until possible reproduction, and stun time
    predatorMask : 2d boolean array
        The locations that contain predators
    plants : 2d scalar array
        The time until plants will be regrown
     : 2d boolean array
        The locations that contain plants
    """
    #- Mask of overlapping prey and predators
    t('1.1.1')
    overlappingMask = preyMask * predatorMask
    overlappingNum = np.count_nonzero(overlappingMask)
    #- Stuns predators
    t('1.1.2')
    #- This technique is faster than generating an entire random array the size
    #- of the grid
    stunnedCheck = np.random.random(overlappingNum) < STUN_CHANCE
    stunnedMask = np.copy(overlappingMask)
    stunnedMask[overlappingMask] = stunnedCheck
    predators[stunnedMask, 2] = STUN_TIME
    t('1.1.3')
    #- Kills predators (predators both stunned and killed will die)
    killCheck = np.random.random(overlappingNum) < PREY_KILL_CHANCE
    killMask = np.copy(overlappingMask)
    killMask[overlappingMask] = killCheck
    predatorMask[killMask] = False
    #- Eats prey (if the predator was not stunned or killed)
    t('1.1.4')
    eatMask = (overlappingMask * np.logical_not( \
        np.logical_or(stunnedMask, killMask)))
    prey[eatMask] = 0
    preyMask[eatMask] = False
    #- Gives predators that ate energy
    t('1.1.5')
    predators[eatMask, 0] = np.where( \
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY < PREDATOR_MAX_ENERGY,
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY, PREDATOR_MAX_ENERGY)
    #- Grows plants that are not fully grown
    t('1.1.6')
    growing = plants > 0
    plants[growing] -= 1
    plantsEaten = (~growing) * preyMask * (prey[:, :, 0] < PREY_HUNGRY)
    #- Resets eaten plants and gives prey energy from eating the plants
    t('1.1.8')
    plants[plantsEaten] = PLANT_REGROWTH_TIME
    prey[plantsEaten, 0] = np.where( \
        prey[plantsEaten, 0] + PREY_EAT_ENERGY < PREY_MAX_ENERGY,
        prey[plantsEaten, 0] + PREY_EAT_ENERGY, PREY_MAX_ENERGY)
    t('1.1.9')
    
def movePredators(preyMask, predators, predatorMask):
    """
    Represents one move cycle for predators

    Parameters
    -------
    preyMask : 2d boolean array
        The locations that contain prey
    predators : 3d scalar array
        First two dimension correspond to position, third dimension is energy,
        time until possible reproduction, and stun time
    predatorMask : 2d boolean array
        The locations that contain predators
    """
    #- Array of indices of predator locations in predatorMask
    predatorIndices = np.argwhere(predatorMask * (predators[:, :, 2] <= 0)) 
    #- Array of indices of prey locations in prey mask
    preyIndices = np.argwhere(preyMask)
    
    #- Check if all predators or prey are dead
    if preyIndices.size == 0 or predatorIndices.size == 0:
        return
    
    #- The number of prey
    num = preyIndices.shape[0]
    #- Creating a copy of each of the prey indices around the map so that
    #  predators can see prey around the borders
    wrappedPreyIndices = np.zeros((preyIndices.shape[0] * 9,
                                   preyIndices.shape[1]), dtype=int)
    wrappedPreyIndices[0:num] =       preyIndices + np.array([-HEIGHT, -WIDTH])
    wrappedPreyIndices[num:num*2] =   preyIndices + np.array([-HEIGHT, 0])
    wrappedPreyIndices[num*2:num*3] = preyIndices + np.array([-HEIGHT, WIDTH])
    wrappedPreyIndices[num*3:num*4] = preyIndices + np.array([0, -WIDTH])
    wrappedPreyIndices[num*4:num*5] = preyIndices + np.array([0, 0])
    wrappedPreyIndices[num*5:num*6] = preyIndices + np.array([0, WIDTH])
    wrappedPreyIndices[num*6:num*7] = preyIndices + np.array([HEIGHT, -WIDTH])
    wrappedPreyIndices[num*7:num*8] = preyIndices + np.array([HEIGHT, 0])
    wrappedPreyIndices[num*8:num*9] = preyIndices + np.array([HEIGHT, WIDTH])
    
    #- If moving diagonally, use euclidean distance
    if DIAGONAL_MOVEMENT:
        distances = cdist(predatorIndices, wrappedPreyIndices, 'euclidean')
    #- Otherwise, use manhattan/cityblock distance
    else:
        distances = cdist(predatorIndices, wrappedPreyIndices, 'cityblock')
    #- If prey are too far from the predator, then use the predator's own
    #  location as a placeholder
    tooFar = np.copy(predatorIndices)
    #- The indexes of the closest prey
    argmin = np.argmin(distances, axis=1)
    #- The locations of the closest prey (or tooFar if they are outside of
    #  eysight)
    closest = np.where(np.min(distances, axis=1)[:,np.newaxis] > \
                       PREDATOR_EYESIGHT, tooFar, wrappedPreyIndices[argmin])
    #- An array to use to move randomly if they are too far away
    randomMovement = np.random.choice([-1, 0, 1], size=closest.shape)
    #- Determine the change in y and change in x for each predator
    dy = np.where(closest[:,0] == predatorIndices[:, 0], randomMovement[:, 0],
                  np.sign(closest[:, 0] - predatorIndices[:, 0]))
    dx = np.where(closest[:,1] == predatorIndices[:, 1], randomMovement[:, 1],
                  np.sign(closest[:, 1] - predatorIndices[:, 1]))
    
    #- The number of predators
    num = predatorIndices.shape[0]
    #- The new indices of the predators, i.e. their old indices + (dy, dx)
    newIndices = predatorIndices + np.reshape(np.vstack((dy, dx)).T, (num, 2))
    #- If there are 0 predators, then return
    if (newIndices.size == 0):
        return
    t('1.3.1')
    #- Wraps around the borders of the screen
    newIndices = np.where(newIndices > -1, np.where(
        newIndices < np.array([HEIGHT, WIDTH]), newIndices,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
    
    #- Prevents predators from overlapping
    #- The number of predators moving to each spot
    counts = np.zeros(newIndices.shape[0], dtype=bool)
    t('1.3.2.1')
    countsEachIndex = {}
    #- For each predator
    for i in range(predatorIndices.shape[0]):
        #- Say "there was a predator here previously"
        #- Need to set it to 1 to differentiate between predators that were
        #  previously there and predators that are moving there
        countsEachIndex[tuple(predatorIndices[i])] = 1
    #- For each predator
    for i in range(newIndices.shape[0]):
        #- Make the indices array into a tuple for hashing
        tup = tuple(newIndices[i])
        #- Basically, check if there was predator with a greater i already at
        #  this location or there is a predator with a lower i moving there
        if tup in countsEachIndex and countsEachIndex[tup]:
            counts[i] = True
        #- Need to set it to 2 to differentiate between predators that were
        #  previously there and predators that are moving there
        countsEachIndex[tup] = 2
        #- Don't have to worry about the old positions of predators with a
        #  lower i value
        if countsEachIndex[tuple(predatorIndices[i])] == 1:
            countsEachIndex[tuple(predatorIndices[i])] = 0
    t('1.3.2.2')
    #- If other predators are moving to the same place, then don't move
    newIndices = np.where(counts[:,np.newaxis], predatorIndices, newIndices)
    
    #- Predators that are staying still, i.e. predators that have the same new indices
    #  as their old indices
    stillIndices = predatorMask[newIndices[:, 0], newIndices[:, 1]]
    #- New indices of moving predators
    newIndices = newIndices[~stillIndices]
    #- Old indices of moving predators, i.e. the old indices of all predators except for
    #  the predators that are staying still
    movedIndices = predatorIndices[~stillIndices]
    #- Move predators
    predators[newIndices[:,0], newIndices[:,1], :] = predators[movedIndices[:,0], movedIndices[:,1], :]
    predators[movedIndices[:,0], movedIndices[:,1], :] = 0
    #- Remove predator's energy
    predators[newIndices[:,0], newIndices[:,1],0] -= PREDATOR_MOVE_ENERGY
    #- Move predators in mask
    predatorMask[newIndices[:,0], newIndices[:,1]] = True
    predatorMask[movedIndices[:,0], movedIndices[:,1]] = False
    #- Kill starving predators
    predatorMask[predatorMask] = np.where(predators[predatorMask, 0] <= 0, False, True)
    predators[predatorMask, 2] -= 1
    t('1.3.2.3')



def movePrey(prey, preyMask, predatorMask, plants):
    """
    Represents one move cycle for prey
    Prey behavior priorities is as follows:
        1. Avoid predators within eyesight
        2. Path towards closest plant cell if hungry
        3. Move in a random direction

    Parameters
    -------
    prey : 3d scalar array
        First two dimension correspond to position, third dimension is energy
        and time until possible reproduction.
    preyMask : 2d boolean array
        The locations that contain prey
    predatorMask : 2d boolean array
        The locations that contain predators
    plants : 2d scalar array
        The locations that contain plants
        The time until plants will be regrown
    """
    #- Array of indices of prey locations in prey mask
    preyIndices = np.argwhere(preyMask)
    #- Array of indices of predator locations in predatorMask
    predatorIndices = np.argwhere(predatorMask) 
    
    #- Check if all predators or prey are dead
    if preyIndices.size == 0 or predatorIndices.size == 0:
        return
    
    #- The number of predators
    num = predatorIndices.shape[0]
    #- Creating a copy of each of the prey indices around the map so that
    #  predators can see predator around the borders
    wrappedPredatorIndices = np.zeros((predatorIndices.shape[0] * 9,
                                   predatorIndices.shape[1]), dtype=int)
    wrappedPredatorIndices[0:num] =       predatorIndices + np.array([-HEIGHT, -WIDTH])
    wrappedPredatorIndices[num:num*2] =   predatorIndices + np.array([-HEIGHT, 0])
    wrappedPredatorIndices[num*2:num*3] = predatorIndices + np.array([-HEIGHT, WIDTH])
    wrappedPredatorIndices[num*3:num*4] = predatorIndices + np.array([0, -WIDTH])
    wrappedPredatorIndices[num*4:num*5] = predatorIndices + np.array([0, 0])
    wrappedPredatorIndices[num*5:num*6] = predatorIndices + np.array([0, WIDTH])
    wrappedPredatorIndices[num*6:num*7] = predatorIndices + np.array([HEIGHT, -WIDTH])
    wrappedPredatorIndices[num*7:num*8] = predatorIndices + np.array([HEIGHT, 0])
    wrappedPredatorIndices[num*8:num*9] = predatorIndices + np.array([HEIGHT, WIDTH])
    
    #- If moving diagonally, use euclidean distance
    if DIAGONAL_MOVEMENT:
        distances = cdist(preyIndices, wrappedPredatorIndices, 'euclidean')
    #- Otherwise, use manhattan/cityblock distance
    else:
        distances = cdist(preyIndices, wrappedPredatorIndices, 'cityblock')
    #- If prey are too far from the predator, then use the predator's own
    #  location as a placeholder
    tooFar = np.array([[0, 0]]*distances.shape[0])
    #- The indexes of the closest prey
    argmin = np.argmin(distances, axis=1)
    #- The locations of the closest prey (or tooFar if they are outside of
    #  eysight)
    closest = np.where(np.min(distances, axis=1)[:,np.newaxis] > \
                       PREY_EYESIGHT, tooFar, wrappedPredatorIndices[argmin])
    #- An array to use to move randomly if they are too far away
    randomMovement = np.random.choice([-1, 1], size=closest.shape)
    #- Determine the change in y and change in x for each prey
    dy = np.where(closest[:,0] == 0, randomMovement[:, 0],
                  np.sign(preyIndices[:, 0] - closest[:, 0]))
    dx = np.where(closest[:,1] == 0, randomMovement[:, 1],
                  np.sign(preyIndices[:, 1] - closest[:, 1]))
    
    if not DIAGONAL_MOVEMENT:
        #- randomly choose either x or y direction to move in, but not both
        if np.random.random() < 0.5:
            dy = np.where(dx == 0, dy, 0)
        else:
            dx = np.where(dy == 0, dx, 0)
    
    #- The number of prey
    num = preyIndices.shape[0]
    #- The new indices of the prey, i.e. their old indices + (dy, dx)
    newIndices = preyIndices + np.reshape(np.vstack((dy, dx)).T, (num, 2))
    newIndices = np.where(np.random.random() < PREY_SPEED, newIndices,
                          preyIndices)
    #- If there are 0 prey, then return
    if (newIndices.size == 0):
        return
    t('1.2.1')
    #- Wraps around the borders of the screen
    newIndices = np.where(newIndices > -1, np.where(
        newIndices < np.array([HEIGHT, WIDTH]), newIndices,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
    
    #- Prevents prey from overlapping
    #- The number of prey moving to each spot
    counts = np.zeros(newIndices.shape[0], dtype=bool)
    t('1.2.2.1')
    countsEachIndex = {}
    #- For each prey
    for i in range(preyIndices.shape[0]):
        #- Say "there was a prey here previously"
        #- Need to set it to 1 to differentiate between prey that were
        #  previously there and prey that are moving there
        countsEachIndex[tuple(preyIndices[i])] = 1
    #- For each prey
    for i in range(newIndices.shape[0]):
        #- Make the indices array into a tuple for hashing
        tup = tuple(newIndices[i])
        #- Basically, check if there was prey with a greater i already at
        #  this location or there is a prey with a lower i moving there
        if tup in countsEachIndex and countsEachIndex[tup]:
            counts[i] = True
        #- Need to set it to 2 to differentiate between prey that were
        #  previously there and prey that are moving there
        countsEachIndex[tup] = 2
        #- Don't have to worry about the old positions of prey with a
        #  lower i value
        if countsEachIndex[tuple(preyIndices[i])] == 1:
            countsEachIndex[tuple(preyIndices[i])] = 0
    #- If other prey are moving to the same place, then don't move
    newIndices = np.where(counts[:,np.newaxis] > 0, preyIndices, newIndices)
    
    #- Prey that are staying still, i.e. prey that have the same new indices
    #  as their old indices
    stillPreyMask = preyMask[newIndices[:, 0], newIndices[:, 1]]
    #- New indices of moving prey
    newIndices = newIndices[~stillPreyMask]
    #- Old indices of moving prey, i.e. the old indices of all prey except for
    #  the prey that are staying still
    movedIndices = preyIndices[~stillPreyMask]
    #- Move prey
    prey[newIndices[:,0], newIndices[:,1], :] = prey[movedIndices[:,0], movedIndices[:,1], :]
    prey[movedIndices[:,0], movedIndices[:,1], :] = 0
    #- Remove prey energy
    prey[newIndices[:,0], newIndices[:,1],0] -= PREY_MOVE_ENERGY
    #- Move prey in mask
    preyMask[newIndices[:,0], newIndices[:,1]] = True
    preyMask[movedIndices[:,0], movedIndices[:,1]] = False
    #- Kill starving prey
    preyMask[preyMask] = np.where(prey[preyMask, 0] <= 0, False, True)
    t('1.2.2')
    
def reproduce(prey, preyMask, predators, predatorMask):
    """
    Represents one reproduction cycle for both predators and prey
    If ASEXUAL reproduction is allowed, then reproduction will occur
    without any check for nearby similar agents, otherwise agents will 
    check for similar agents within their reproduction range.

    Parameters
    -------
    prey : 3d scalar array
        First two dimension correspond to position, third dimension is energy
        and time until possible reproduction.
    preyMask : 2d boolean array
        The locations that contain prey
    predatorMask : 2d boolean array
        The locations that contain predators
    plants : 2d scalar array
        The locations that contain plants
        The time until plants will be regrown
    """
    #- Array of indices of predator locations
    predatorIndices = np.argwhere(predatorMask)
    #- Decrease the time till predators can reproduce again
    predators[predatorIndices[:, 0], predatorIndices[:, 1], 1] -= 1
    #- Calculates the distance to the nearest predator if it can't reproduce
    #  asexually
    if not ASEXUAL:
        #- If moving diagonally, then use euclidean distance
        if DIAGONAL_MOVEMENT:
            distances = cdist(predatorIndices, predatorIndices, 'euclidean')
        #- Otherwise, use manhattan/cityblock distance
        else:
            distances = cdist(predatorIndices, predatorIndices, 'cityblock')
        #- The distance to itself is always 0, so ignore that value by setting
        #  it to somethign above the reproduction range
        distances = np.where(distances == 0, PREDATOR_REPRODUCTION_RANGE + 1,
                             distances)
        #- Determine which distance are within the necessary range
        inRange = np.any(distances < PREDATOR_REPRODUCTION_RANGE, axis=1)
    else:
        #- If it can reproduce asexually, all distances are within range
        inRange = np.ones(predatorIndices.shape[0], dtype=bool)
    #- Has it been long enough since the predator last reproduced?
    canReproduceTime = predators[predatorIndices[:, 0],
                                  predatorIndices[:, 1], 1] <= 0
    #- Does the predator have enough food?
    canReproduceFood = predators[predatorIndices[:, 0],
                                  predatorIndices[:, 1], 0] >= \
                                      PREDATOR_REPRODUCTION_THRESHOLD
    #- Combine those three to see if it can reproduce
    canReproduce = canReproduceTime * canReproduceFood * inRange
    #- Get the indexes of predators that can reproduce
    reproducingIndices = predatorIndices[canReproduce]
    #- Neighbors represent a 3x3 grid of all the neighbors of each index:
    #  0  1  2
    #  3  X  4
    #  5  6  7
    neighbors = np.zeros((reproducingIndices.shape[0], 8, 2), dtype=int)
    neighbors[:, :, :] = reproducingIndices[:, np.newaxis, :]
    neighbors[:, 0] += -1
    neighbors[:, 1, 0] += -1
    #- Fancy way of subtracting -1 from y and adding 1 to x
    neighbors[:, 2] += np.array([-1, 1])[np.newaxis, :]
    neighbors[:, 3, 1] += -1
    neighbors[:, 4, 1] += 1
    neighbors[:, 5] += np.array([1, -1])[np.newaxis, :]
    neighbors[:, 6, 0] += 1
    neighbors[:, 7] += 1
    
    #- Wraps around the borders of the screen
    neighbors = np.where(neighbors > -1, np.where(
        neighbors < np.array([HEIGHT, WIDTH]), neighbors,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
    
    #- Determines which neighboring cells are occupied
    viableNeighbors = np.all(np.isin(neighbors, predatorIndices), axis=2)
    #- Fills those bad cells with -1
    neighbors[~viableNeighbors] = -1
    
    #- Creates an empty list for the new locations of each prey/predator
    newLocations = []
    #- Loops through each new child
    for i in range(neighbors.shape[0]):
        #- Determiens which neighbors are viable
        viableNeighbors = neighbors[i][neighbors[i] != -1]
        #- If there are no viable neighbors, then try again next timestep
        if viableNeighbors.size == 0:
            continue
        #- Reshape this into indices
        viableNeighbors = np.reshape(viableNeighbors,
                                     (int(viableNeighbors.size / 2), 2))
        #- Randomly generate an index to choose
        index = np.random.randint(0, viableNeighbors.shape[0])
        #- Add that location to newLocations
        newLocations.append(viableNeighbors[index])
    
    #- Make newLocations an array
    newLocations = np.array(newLocations, dtype=int)
    #- Reshape newLocations into indices
    newLocations = np.reshape(newLocations, (int(newLocations.size / 2), 2))
    
    #- Spawn the new predator in the mask
    predatorMask[newLocations[:, 0], newLocations[:, 1]] = True
    #- Give all the appropriate start values (energy, reproduction time, etc.)
    predators[newLocations[:, 0], newLocations[:, 1], 0] = \
        PREDATOR_START_ENERGY
    predators[newLocations[:, 0], newLocations[:, 1], 1] = random.randint( \
        PREDATOR_REPRODUCTION_TIME[0], PREDATOR_REPRODUCTION_TIME[1])
    #- Set the old predator's reproduction time back to the start amount
    predators[reproducingIndices[:, 0], reproducingIndices[:, 1],1] = \
        random.randint(PREDATOR_REPRODUCTION_TIME[0],
                       PREDATOR_REPRODUCTION_TIME[1])
    t('1.9.1')
    
                
    #- Array of indices of prey locations
    preyIndices = np.argwhere(preyMask)
    #- Decrease the time till prey can reproduce again
    prey[preyIndices[:, 0], preyIndices[:, 1], 1] -= 1
    #- Calculates the distance to the nearest prey if it can't reproduce
    #  asexually
    if not ASEXUAL:
        #- If moving diagonally, then use euclidean distance
        if DIAGONAL_MOVEMENT:
            distances = cdist(preyIndices, preyIndices, 'euclidean')
        #- Otherwise, use manhattan/cityblock distance
        else:
            distances = cdist(preyIndices, preyIndices, 'cityblock')
        #- The distance to itself is always 0, so ignore that value by setting
        #  it to somethign above the reproduction range
        distances = np.where(distances == 0, PREY_REPRODUCTION_RANGE + 1,
                             distances)
        #- Determine which distance are within the necessary range
        inRange = np.any(distances < PREY_REPRODUCTION_RANGE, axis=1)
    else:
        #- If it can reproduce asexually, all distances are within range
        inRange = np.ones(preyIndices.shape[0], dtype=bool)
    #- Has it been long enough since the prey last reproduced?
    canReproduceTime = prey[preyIndices[:, 0],
                                  preyIndices[:, 1], 1] <= 0
    #- Does the prey have enough food?
    canReproduceFood = prey[preyIndices[:, 0],
                                  preyIndices[:, 1], 0] >= \
                                      PREY_REPRODUCTION_THRESHOLD
    #- Combine those three to see if it can reproduce
    canReproduce = canReproduceTime * canReproduceFood * inRange
    #- Get the indexes of prey that can reproduce
    reproducingIndices = preyIndices[canReproduce]
    
    #- Neighbors represent a 3x3 grid of all the neighbors of each index:
    #  0  1  2
    #  3  X  4
    #  5  6  7
    neighbors = np.zeros((reproducingIndices.shape[0], 8, 2), dtype=int)
    neighbors[:, :, :] = reproducingIndices[:, np.newaxis, :]
    neighbors[:, 0] += -1
    neighbors[:, 1, 0] += -1
    #- Fancy way of subtracting -1 from y and adding 1 to x
    neighbors[:, 2] += np.array([-1, 1])[np.newaxis, :]
    neighbors[:, 3, 1] += -1
    neighbors[:, 4, 1] += 1
    neighbors[:, 5] += np.array([1, -1])[np.newaxis, :]
    neighbors[:, 6, 0] += 1
    neighbors[:, 7] += 1
    
    #- Wraps around the borders of the screen
    neighbors = np.where(neighbors > -1, np.where(
        neighbors < np.array([HEIGHT, WIDTH]), neighbors,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
    
    #- Determines which neighboring cells are occupied
    unViableNeighbors = np.all(np.isin(neighbors, preyIndices), axis=2)
    #- Fills those bad cells with -1
    neighbors[unViableNeighbors] = -1
    
    #- Creates an empty list for the new locations of each prey/prey
    newLocations = []
    #- Loops through each new child
    for i in range(neighbors.shape[0]):
        #- Determiens which neighbors are viable
        viableNeighbors = neighbors[i][neighbors[i] != -1]
        #- If there are no viable neighbors, then try again next timestep
        if viableNeighbors.size == 0:
            continue
        #- Reshape this into indices
        viableNeighbors = np.reshape(viableNeighbors,
                                     (int(viableNeighbors.size / 2), 2))
        #- Randomly generate an index to choose
        index = np.random.randint(0, viableNeighbors.shape[0])
        #- Add that location to newLocations
        newLocations.append(viableNeighbors[index])
    
    #- Make newLocations an array
    newLocations = np.array(newLocations, dtype=int)
    #- Reshape newLocations into indices
    newLocations = np.reshape(newLocations, (int(newLocations.size / 2), 2))
    
    #- Spawn the new prey in the mask
    preyMask[newLocations[:, 0], newLocations[:, 1]] = True
    #- Give all the appropriate start values (energy, reproduction time, etc.)
    prey[newLocations[:, 0], newLocations[:, 1], 0] = \
        PREY_START_ENERGY
    prey[newLocations[:, 0], newLocations[:, 1], 1] = random.randint( \
        PREY_REPRODUCTION_TIME[0], PREY_REPRODUCTION_TIME[1])
    #- Set the old prey's reproduction time back to the start amount
    prey[reproducingIndices[:, 0], reproducingIndices[:, 1],1] = \
        random.randint(PREY_REPRODUCTION_TIME[0],
                       PREY_REPRODUCTION_TIME[1])
    t('1.9.2')
    
#- Dictionary containing overall times for each action
times = {}
#- Tracks the previous timestamp when time was measured
lastTime = time.time()
#- The number of total timesteps
timesteps = 0

def t(label):
    """
    Tracks the amount of time since this function was last called and adds it
    to a dictionary. If the same label has been used before (for example,
    through looping), then this time is added on to that.

    Parameters
    ----------
    label : string
        The label for this time, usually following 'x.y.z' format. If this is
        labeled "TIMESTEP" then it is instead a counter to say that a timestep
        has passed.
    """
    global timesteps
    global lastTime
    if label == "TIMESTEP":
        timesteps += 1
        return
    curTime = time.time()
    #- If this section has already been added to the dictionary:
    if label in times:
        times[label] += curTime - lastTime
    #- Otherwise, add it to the dictionary:
    else:
        times[label] = curTime - lastTime
    lastTime = curTime

def printTimes():
    """
    Performs several operations on the values in the times dictionary and
    prints them out.
    First, converts each amount of time to a percentage of the total time used.
    Also removes the X label from the dictionary, which represents anything
    that could throw off the overall percentage (like visualization).
    Next, rounds the values to a certain number of decimal places.
    Finally, indents the lines to line up evenly.
    """
    #- The number of decimal places to round to
    decimalPlaces = 5
    #- This value is usually much larger than others and throws off the rest
    if 'X' in times:
        del times['X']
    #- Max length of any label (used for indentation)
    maxLength = 0
    #- Used to calculate percentage
    total = np.sum(np.array(list(times.values())))
    #- Finds the maximum length of the labels
    for name in times.keys():
        if len(name) > maxLength:
            maxLength = len(name)
    print(f"Average time per timestep: {total/timesteps} seconds")
    for name, timeAmount in times.items():
        #- Commented out code uses sig figs instead of decimal places
        #magnitude = math.floor(math.log10(abs(timeAmount / total))) + 3
        #value = round(timeAmount / total * 100, sigFigs - magnitude)
        #- Converts values to a percentage and rounds them
        value = round(timeAmount / total * 100, decimalPlaces)
        indentation = maxLength - len(name)
        indentationString = indentation * " "
        #- Indents values below ten slightly more to align places
        #- Don't have to worry about values above 100 because it's a percent
        if value < 10:
            indentationString += " "
        print(f"{name}:{indentationString} {value}%")

#- Checks if this file is being run directly and not imported
if (__name__ == '__main__'):
    np.set_printoptions(threshold=sys.maxsize)
    preyPopulations, predatorPopulations = runSimulation(True)


