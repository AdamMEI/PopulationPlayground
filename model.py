#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:58:49 2023

@author: adamthepig
"""

#------------------------------- MODULE IMPORTS -------------------------------

import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import time
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
FPS = 100
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
PREDATOR_START_NUM = 50
#- Starting Energy Value
PREDATOR_START_ENERGY = 50
#- Energy Gain by Eating 
PREDATOR_EAT_ENERGY = 50
#- Maximum Energy Value
PREDATOR_MAX_ENERGY = 100
#- Reproduction Timer (How many cycles between possible reproductions)
PREDATOR_REPRODUCTION_TIME = (10, 20)
#- Eyesight Radius
PREDATOR_EYESIGHT = 10
#- Energy Spent in Single Move Cycle
PREDATOR_MOVE_ENERGY = 1
#- Number of Cycles stunned
STUN_TIME = 5
#- Radius that another predator has to be in for reproduction to occure
PREDATOR_REPRODUCTION_RANGE = 5
#- Energy threshold for reproduction
PREDATOR_REPRODUCTION_THRESHOLD = 80

#- Prey Constants
#-----------------
#- Starting Number of Agents
PREY_START_NUM = 50
#- Starting Energy Value
PREY_START_ENERGY = 50
#- Energy Gain by Eating 
PREY_EAT_ENERGY = 5
#- Maximum Energy Value
PREY_MAX_ENERGY = 100
#- Reproduction Timer (How many cycles between possible reproductions)
PREY_REPRODUCTION_TIME = (20, 30)
#- Eyesight Radius
PREY_EYESIGHT = 7
#- Energy Spent in Single Move Cycle
PREY_MOVE_ENERGY = 1
#- Energy Threshold for Eating
PREY_HUNGRY = 90
#- Percent chance of stunning predators
STUN_CHANCE = 0.5
#- Percent chance of killing predators
PREY_KILL_CHANCE = 0.01
#- Radius that another prey has to be in for reproduction to occure
PREY_REPRODUCTION_RANGE = 5
#- Energy threshold for reproduction
PREY_REPRODUCTION_THRESHOLD = 70

#- Plant Constants
#------------------
#- The time required for plants to regrow after being eaten
PLANT_REGROWTH_TIME = 250

#- Analysis Constants
#------------------
#- Whether each plot should be displayed after a simulation, set, or set of
#  sets.
LAG_PLOT_DISLPAYS = {"sets": True, "set": True, "simulation": False}
POPULATION_PLOT_DISPLAYS = {"sets": True, "set": True, "simulation": False}

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
        print(f'Running...{i}', end='\r')
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
preyPopulations[i + lag] and predatorPopulations[i - lag]")
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
    #- Checks to see if the window has been closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    #- Iterates through grid
    for i in range(WIDTH):
        for j in range(HEIGHT):
            #- Draws a rectangle for each prey, predator, and plant
            rect = pygame.Rect(i * PIXEL_SIZE, j * PIXEL_SIZE,
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
    #- lists for keeping track of changes in x and y for each predator
    dx = []
    dy = []
    #- loop through all predators
    for i in range(len(predatorIndices)):
        #- (x, y) coordinate of predator in predatorMask
        x, y = predatorIndices[i,0], predatorIndices[i,1]

        #- eyesightArray gives us an array with (x,y) as the center and a radius of predator eyesight
        #- eyesight will contain any indices of prey within the eyesightArray
        eyesight = np.argwhere(eyesightArray(preyMask, x, y, PREDATOR_EYESIGHT))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREDATOR_EYESIGHT, PREDATOR_EYESIGHT)
        #- Calculates relative distance to all prey within eyesight from x,y coordinate of predator
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
        else:
            #- Manhattan distance
            distances = np.sum(np.abs(eyesight-center), axis=1)
        #- no prey within eyesight
        if len(distances) == 0:
            #- random movement in x and y direction
            dx.append(np.random.choice([0,1]))
            dy.append(np.random.choice([-1,0,1]))
        #- atleast 1 prey within eyesight
        else:
            #- Index of closest prey(s)
            closestPreyIndex = np.where(distances == distances.min())
            #- Randomly choose closest prey if there is more than one option
            if len(closestPreyIndex[0]) > 1: closestPreyIndex = np.random.choice(closestPreyIndex[0])
            #- Predators have a chance to move one extra space
            boost = np.random.binomial(1,0.20)
            #- Move towards the closest prey in both x and y direction
            if eyesight[closestPreyIndex,0] > center[0]: dx.append(1+boost)
            elif eyesight[closestPreyIndex,0] < center[0]: dx.append(-1-boost)
            else: dx.append(0)
            if eyesight[closestPreyIndex,1] > center[1]: dy.append(1+boost)
            elif eyesight[closestPreyIndex,1] < center[1]: dy.append(-1-boost)
            else: dy.append(0)
    dx = np.asarray(dx)
    dy = np.asarray(dy)

    if not DIAGONAL_MOVEMENT:
        #- randomly choose either x or y direction to move in, but not both
        if np.random.random() < 0.5:
            dy = np.where(dx == 0, dy, 0)
        else:
            dx = np.where(dy == 0, dx, 0)
            
    #- The number of predators
    num = predatorIndices.shape[0]
    #- The new indices of the predators, i.e. their old indices + (dy, dx)
    newIndices = predatorIndices + np.reshape(np.vstack((dx, dy)).T, (num, 2))
    #- If there are 0 predators, then return
    if (newIndices.size == 0):
        return
    t('1.3.1')
    #- Wraps around the borders of the screen
    newIndices = np.where(newIndices > -1, np.where(
        newIndices < np.array([HEIGHT, WIDTH]), newIndices,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
    #- predators that are staying still, i.e. predators that have the same new indices
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
    t('1.3.2')
       
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
    #- Array of indices of prey locations
    preyIndices = np.argwhere(preyMask)
    #- lists for keeping track of changes in x and y for each prey
    dx = []
    dy = []
    #- loop through all prey
    for i in range(len(preyIndices)):
        #- (x, y) coordinate of prey in preyMask
        x, y = preyIndices[i,0], preyIndices[i,1]
        #- eyesightArray gives us an array with (x,y) as the center and a radius of prey eyesight
        #- plantEyesight will contain any indices of plants within the eyesightArray
        #- predatorEyesight will contain any indices of predators within the eyesightArray
        # plantEyesight = np.transpose(np.where(eyesightArray(plants, x, y, PREY_EYESIGHT) == 0))  
        predatorEyesight = np.argwhere(eyesightArray(predatorMask,x,y,PREY_EYESIGHT))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREY_EYESIGHT, PREY_EYESIGHT)
        #- Calculates relative distance to all predators and plants from x,y coordinate of prey
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            predDistances = np.floor(np.sqrt(np.sum((predatorEyesight-center)**2, axis=1)))
            # plantDistances = np.floor(np.sqrt(np.sum((plantEyesight-center)**2, axis=1)))
        else:
            #- Manhattan distance
            predDistances = np.sum(np.abs(predatorEyesight-center), axis=1)
            # plantDistances = np.sum(np.abs(plantEyesight-center), axis=1)
        #- no predators within eyesight
        if len(predDistances) == 0:
            #- no plants within eyesight or prey is not hungry
            # if len(plantDistances) <= 1 or prey[x,y,0] > PREY_HUNGRY:
                #- random movement in x and y direction
            dx.append(np.random.choice([-1,0,1]))
            dy.append(np.random.choice([-1,0,1]))
            # #- atleast 1 plant within eyesight
            # else:
            #     #- Index of closest plant(s)
            #     if np.size(plantDistances) <= 1:
            #         closestPlantIndex = np.array(((center[0], center[1])))
            #     else:
            #         closestPlantIndex = np.where(plantDistances == plantDistances[np.nonzero(plantDistances)].min())
            #     #- Randomly choose closest plant if there is more than one option
            #     if len(closestPlantIndex[0]) > 1: closestPlantIndex = np.random.choice(closestPlantIndex[0])
            #     else: closestPlantIndex = closestPlantIndex[0][0]
            #     #- Move towards closest plant in both x and y direction
            #     if plantEyesight[closestPlantIndex,0] > center[0]: dx.append(1)
            #     elif plantEyesight[closestPlantIndex,0] < center[0]: dx.append(-1)
            #     else: dx.append(0)
            #     if plantEyesight[closestPlantIndex,1] > center[1]: dy.append(1)
            #     elif plantEyesight[closestPlantIndex,1] < center[1]: dy.append(-1)
            #     else: dy.append(0)
        #- atleast 1 predator within eyesight
        else:
            #- Index of closest predator(s)
            closestPredatorIndex = np.where(predDistances == predDistances.min())
            #- Randomly choose closest predator if there is more than one option
            if len(closestPredatorIndex[0]) > 1: closestPredatorIndex = np.random.choice(closestPredatorIndex[0])
            #- Move towards closest predator in both x and y direction
            else: closestPredatorIndex = closestPredatorIndex[0][0]
            if predatorEyesight[closestPredatorIndex,0] > center[0]: dx.append(-1)
            elif predatorEyesight[closestPredatorIndex,0] < center[0]: dx.append(1)
            else: dx.append(0)
            if predatorEyesight[closestPredatorIndex,1] > center[1]: dy.append(-1)
            elif predatorEyesight[closestPredatorIndex,1] < center[1]: dy.append(1)
            else: dy.append(0)
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    
    if not DIAGONAL_MOVEMENT:
        #- randomly choose either x or y direction to move in, but not both
        if np.random.random() < 0.5:
            dy = np.where(dx == 0, dy, 0)
        else:
            dx = np.where(dy == 0, dx, 0)
    
    #- The number of prey
    num = preyIndices.shape[0]
    #- The new indices of the prey, i.e. their old indices + (dy, dx)
    newIndices = preyIndices + np.reshape(np.vstack((dx, dy)).T, (num, 2))
    #- If there are 0 prey, then return
    if (newIndices.size == 0):
        return
    t('1.2.1')
    #- Wraps around the borders of the screen
    newIndices = np.where(newIndices > -1, np.where(
        newIndices < np.array([HEIGHT, WIDTH]), newIndices,
       0), np.array([HEIGHT - 1, WIDTH - 1]))
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
    
def eyesightArray(a, x, y, n):
    """
    Given an array a, returns an array with radius n with center element a[x,y]
    This is used to assist in consistent calculations across periodic boundary
    conditions
     
    Source: https://stackoverflow.com/questions/4148292/how-do-i-select-a-window-from-a-numpy-array-with-periodic-boundary-conditions
    
    Parameters
    -------
    a : array-like
    x : int
        x index in array a
    y : int 
        y index in array a
    n : int
        radius of desired array
        
    Returns
    -------
    Subarray of input array with radius n with center element a[x,y] 
    """
    n = (n*2) + 1
    a=np.roll(np.roll(a,shift=-x+int(n/2),axis=0),shift=-y+int(n/2),axis=1)
    return a[:n,:n]
    
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
    #- Loop through all predators
    for i in range(len(predatorIndices)):
        #- similar agents in range flag
        inRange = True
        #- x,y indices of current predator in predators array
        x, y = predatorIndices[i,0], predatorIndices[i,1]
        #- eyesightArray gives us an array with (x,y) as the center and a radius of predator eyesight
        #- eyesight will contain any indices of predators within the eyesightArray
        eyesight = np.argwhere(eyesightArray(predatorMask, x, y, PREDATOR_REPRODUCTION_RANGE))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREDATOR_REPRODUCTION_RANGE, PREDATOR_REPRODUCTION_RANGE)
        #- similar agents required within range for reproduction to happen
        if not ASEXUAL:
            if DIAGONAL_MOVEMENT:
                #- Euclidean distance
                distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
            else:
                #- Manhattan distance
                distances = np.sum(np.abs(eyesight-center), axis=1)
            #- True when asexual reproduction, false when no similar agents within range
            inRange = not len(distances) == 0
            
        #- no similar agents within range or energy is less than threshold or agent has reproduced recently
        if not inRange or predators[x,y,0] < PREDATOR_REPRODUCTION_THRESHOLD or predators[x,y,1] > 0:
            #- decrease reproduction timer
            predators[x,y,1] -= 1
            continue
        else:
            #- get list of neighbor cells (x, y) coordinates to current predator
            neighbors = getNeighbors(x, y, predatorMask)
            #- randomly pick a valid neighbor and spawn new agent there
            newx, newy = neighbors[np.random.choice(len(neighbors))]
            predatorMask[newx, newy] = True
            predators[newx, newy, 0] = PREDATOR_START_ENERGY
            predators[newx, newy, 1] = random.randint( \
                PREDATOR_REPRODUCTION_TIME[0], PREDATOR_REPRODUCTION_TIME[1])
            predators[x,y,1] = random.randint( \
                PREDATOR_REPRODUCTION_TIME[0], PREDATOR_REPRODUCTION_TIME[1])
                
    #- Array of indices of prey locations
    preyIndices = np.argwhere(preyMask)
    #- Loop through all prey
    for i in range(len(preyIndices)):
        #- similar agents in range flag
        inRange = True
        #- x,y indices of current predator in predators array
        x, y = preyIndices[i,0], preyIndices[i,1]
        #- eyesightArray gives us an array with (x,y) as the center and a radius of prey eyesight
        #- eyesight will contain any indices of prey within the eyesightArray
        eyesight = np.argwhere(eyesightArray(preyMask, x, y, PREY_REPRODUCTION_RANGE))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREY_REPRODUCTION_RANGE, PREY_REPRODUCTION_RANGE)
        #- similar agents required within range for reproduction to happen
        if not ASEXUAL:
            if DIAGONAL_MOVEMENT:
                #- Euclidean distance
                distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
            else:
                #- Manhattan distance
                distances = np.sum(np.abs(eyesight-center), axis=1)
            #- True when asexual reproduction, false when no similar agents within range
            inRange = not len(distances) == 0
            
        #- no similar agents within range or energy is less than threshold or agent has reproduced recently
        if not inRange or prey[x,y,0] < PREY_REPRODUCTION_THRESHOLD or prey[x,y,1] > 0:
            #- decrease reproduction timer
            prey[x,y,1] -= 1
            continue
        else:
            #- get list of neighbor cells (x, y) coordinates to current prey
            neighbors = getNeighbors(x, y, preyMask)
            #- randomly pick a valid neighbor and spawn new agent there
            newx, newy = neighbors[np.random.choice(np.arange(len(neighbors)))]
            preyMask[newx, newy] = True
            prey[newx, newy, 0] = PREDATOR_START_ENERGY
            prey[newx, newy, 1] = random.randint( \
                PREY_REPRODUCTION_TIME[0], PREY_REPRODUCTION_TIME[1])
            prey[x,y,1] = random.randint( \
                PREY_REPRODUCTION_TIME[0], PREY_REPRODUCTION_TIME[1])

def getNeighbors(x, y, mask=None):
    """
    Returns a list of indice tuples for neighboring cells of (x,y), optionally 
    can be provided a mask to remove occupied neighboring cells from the returned list
    
    Parameters
    -------
    x : int
        x indice in an array
    y : int 
        y indice in an array
    mask : boolean array
        mask to check for occupied neighbors
        
    Returns
    -------
    list of indice tuples
    """
    #- get list of neighbor cells (x, y) coordinates to current predator
    neighbors = [(x-1, y-1),(x, y-1),(x+1, y-1),(x-1, y),(x+1, y),(x-1, y+1),(x, y+1),(x+1, y+1)]
    #- loop through neighbors to adjust values for boundary conditions
    for i in range(len(neighbors)):
        #- check for going off both bottom and right side of screen
        if neighbors[i][0] >= WIDTH and neighbors[i][1] >= HEIGHT:
            neighbors[i] = (0, 0)
        #- check for going off right side of screen
        elif neighbors[i][0] >= WIDTH:
            neighbors[i] = (0, neighbors[i][1])
        #- check for going off bottom side of screen
        elif neighbors[i][1] >= HEIGHT:
            neighbors[i] = (neighbors[i][0], 0)
    #- checks mask against neighbors and removes occupied neighbor cells
    if mask is not None:
        for neighbor in neighbors:
            if mask[neighbor[0],neighbor[1]]:
                neighbors.remove(neighbor)
    #- if no valid neighbors, return original cell as only valid location
    if len(neighbors) == 0:
        return [(x,y)]
    return neighbors
    
#- Dictionary containing overall times for each action
times = {}
#- Tracks the previous timestamp when time was measured
lastTime = time.time()

def t(label):
    """
    Tracks the amount of time since this function was last called and adds it
    to a dictionary. If the same label has been used before (for example,
    through looping), then this time is added on to that.

    Parameters
    ----------
    label : string
        The label for this time, usually following 'x.y.z' format
    """
    global lastTime
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
    runSimulation(True)


