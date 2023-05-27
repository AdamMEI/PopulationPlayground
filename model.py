#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:58:49 2023

@author: adamthepig
"""

#------------------------------- MODULE IMPORTS -------------------------------

import numpy as np
import math
import random
import pygame
import matplotlib.pyplot as plt
import time
#--------------------------------- CONSTANTS ----------------------------------

#- Number of sets of simulations run
SET_NUM = 10
#- Number of simulations run per set
SIMULATION_NUM = 10
#- Number of time steps in the simulation
TIME_STEPS = 1000
#- Grid size
m = 100 #- Horizontal
n = 100 #- Vertical
#- The starting number of prey, predators, and plants
PREY_START_NUM = 80
PREDATOR_START_NUM = 50
#- The energy that prey and predators start with
PREY_START_ENERGY = 50
PREDATOR_START_ENERGY = 100
#- The amount of energy gained by prey and predators after eating
PREY_EAT_ENERGY = 10
PREDATOR_EAT_ENERGY = 50
#- The maximum energy that prey and predators can have
PREY_MAX_ENERGY = 100
PREDATOR_MAX_ENERGY = 100
#- The amount of time until prey and predators can reproduce for the first time
PREY_REPRODUCTION_START_TIME = 30
PREDATOR_REPRODUCTION_START_TIME = 2
#- The time required for plants to regrow after being eaten
PLANT_REGROWTH_TIME = 100
#- The chance of prey fighting back and stunning or killing predators
STUN_CHANCE = 0.5
PREY_KILL_CHANCE = 0.01
#- The amount of time that predators are stunned for after prey stun them
STUN_TIME = 5
#- The number of pixels for each grid location
PIXEL_SIZE = 7
#- The visualization colors
BACKGROUND_COLOR = (255, 255, 255)
PREY_COLOR = (0, 0, 255)
PREDATOR_COLOR = (255, 0, 0)
#- Plant colors are interpolated between these two
PLANT_GROWN_COLOR = (0, 255, 0)
PLANT_UNGROWN_COLOR = (255, 255, 0)
#- FPS of visualization
FPS = 60

#- Predator and Prey Eyesight values
PREDATOR_EYESIGHT = 20
PREY_EYESIGHT = 10
#- Predator and Prey Movement Energy Costs
PREDATOR_MOVE_ENERGY = 1
PREY_MOVE_ENERGY = 1
#- Predator and Prey Hungry Threshold Values
PREDATOR_HUNGRY = 90
PREY_HUNGRY = 90
#- Range and energy threshold for Reproduction
REPRODUCTION_RANGE = 5
REPRODUCTION_THRESHOLD = 80
#- Can agents move diagonally? (affects distance calc)
DIAGONAL_MOVEMENT = True

def runSets():
    """
    Runs multiple sets of simulations, then prints average populations at the
    end of the simulation and the standard deviation of the populations.
    """
    preyPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    predatorPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    for i in range(SET_NUM):
        print(f'Running Sets...{i+1}', end='\r')
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
    for i in range(SIMULATION_NUM):
        preyPopulations[i], predatorPopulations[i] = runSimulation(False)
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
    prey, preyMask, predators, predatorMask, plants,  = initialize()
    if shouldVisualize:
        screen = initVisualization()
    t('X')
    for i in range(TIME_STEPS):
        #- stop simulation if there are no prey or no predators alive
        if np.any(preyMask) and np.any(predatorMask):
            if shouldVisualize:
                screen.fill((255,255,255))
                #- Visualizes the grid
                if not visualize(screen, preyMask, predatorMask,
                                 plants):
                    break
            t('X')
            #- Runs a single feed cycle
            t('1.1')
            feed(prey, preyMask, predators, predatorMask, plants)
            t('1.2')
            #- Runs a single move cycle
            movePredators(preyMask, predators, predatorMask)
            movePrey(prey, preyMask, predators, predatorMask, plants)
            reproduce(prey, preyMask, predators, predatorMask)
            
            preyPopulations[i] = np.count_nonzero(preyMask)
            predatorPopulations[i] = np.count_nonzero(predatorMask)
            t('1.4')
        else:
            break
    if shouldVisualize:
        pygame.quit()
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
    prey = np.zeros((m, n, 2))
    preyMask = np.zeros((m, n), dtype=bool)
    predators = np.zeros((m, n, 3))
    predatorMask = np.zeros((m, n), dtype=bool)
    #- Create masks
    i = 0
    while i < PREY_START_NUM:
        x = random.randint(1, n - 2)
        y = random.randint(1, m - 2)
        #- Ensure that there is not already a prey there
        if not preyMask[y, x]:
            preyMask[y, x] = True
            i += 1
    i = 0
    while i < PREDATOR_START_NUM:
        x = random.randint(1, n - 2)
        y = random.randint(1, m - 2)
        #- Ensure that there is not already a prey or predator there
        if not predatorMask[y, x] and not preyMask[y, x]:
            predatorMask[y, x] = True
            i += 1
    #- Sets prey and predator start values
    prey[preyMask, 0] = PREY_START_ENERGY
    prey[preyMask, 1] = PREY_REPRODUCTION_START_TIME
    predators[predatorMask, 0] = PREDATOR_START_ENERGY
    predators[predatorMask, 1] = PREDATOR_REPRODUCTION_START_TIME
    #- Initializes plants
    plants = np.zeros((m, n))
    return (prey, preyMask, predators, predatorMask, \
            plants)

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
    screen = pygame.display.set_mode([m * PIXEL_SIZE, n * PIXEL_SIZE])
    #- Sets background color
    screen.fill((255, 255, 255))
    return screen

def visualize(screen, preyMask, predatorMask, plants):
    """
    Visualizes the prey, predators, and plants using the constant colors and
    constant background color. Also uses PIXEL_SIZE and the grid size given by
    m and n.

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
    for i in range(m):
        for j in range(n):
            #- Draws a rectangle for each prey, predator, and plant
            rect = pygame.Rect(i * PIXEL_SIZE, j * PIXEL_SIZE,
                               PIXEL_SIZE, PIXEL_SIZE)
            if preyMask[i, j]:
                pygame.draw.rect(screen, PREY_COLOR, rect)
            elif predatorMask[i, j]:
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
    grownOverlapping = (~growing) * preyMask
    #- Resets eaten plants and gives prey energy from eating the plants
    t('1.1.8')
    plants[grownOverlapping] = PLANT_REGROWTH_TIME
    prey[grownOverlapping, 0] = np.where( \
        prey[grownOverlapping, 0] + PREY_EAT_ENERGY < PREY_MAX_ENERGY,
        prey[grownOverlapping, 0] + PREY_EAT_ENERGY, PREY_MAX_ENERGY)
    t('1.1.9')
    
def movePredators(preyMask, predators, predatorMask):
    """
    Represents one move cycle for predators

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
    #- Array of indices of predator and prey locations
    predatorIndices = np.argwhere(predatorMask)  
    #- Find index of closest prey within eyesight for each predator
    dx = []
    dy = []
    for i in range(len(predatorIndices)):
        #- (x, y) coordinate of predator in predatorMask
        x, y = predatorIndices[i,0], predatorIndices[i,1]
        #- eyesightArray gives us an array with (x,y) as the center and a radius of predator eyesight
        #- eyesight will contain any indices of prey within the eyesightArray
        eyesight = np.argwhere(eyesightArray(preyMask, x, y, PREDATOR_EYESIGHT*2+1))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREDATOR_EYESIGHT, PREDATOR_EYESIGHT)
        #- Calculates relative distance to all prey within eyesight from x,y coordinate of predator
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
        else:
            #- Manhattan distance
            distances = np.sum(np.abs(eyesight-center), axis=1)
        #- check if there is no prey within eyesight (true if the entire array is masked)
        if len(distances) == 0:
            dx.append(np.random.choice([-1,0,1]))
            dy.append(np.random.choice([-1,0,1]))
        else:
            closestPreyIndex = np.argmin(distances)
            boost = np.random.binomial(1,0.25)
            if eyesight[closestPreyIndex,0] > center[0]: dx.append(1+boost)
            elif eyesight[closestPreyIndex,0] < center[0]: dx.append(-1-boost)
            else: dx.append(0)
            if eyesight[closestPreyIndex,1] > center[1]: dy.append(1+boost)
            elif eyesight[closestPreyIndex,1] < center[1]: dy.append(-1-boost)
            else: dy.append(1)
    dx = np.asarray(dx)
    dy = np.asarray(dy)

    if not DIAGONAL_MOVEMENT:
        #- prioritize moving in x direction first (move in y once x coordinates line up)
        dy = np.where(dx == 0, dy, 0)
    
    #- Update predator and predatorMask values for movement
    for i in range(len(predatorIndices)):
        x, y = predatorIndices[i,0], predatorIndices[i,1]
        #- if moving causes energy to equal or go below 0, predator dies
        if (predators[x, y, 0] - PREDATOR_MOVE_ENERGY <= 0):
                predators[x, y, 0] = 0
                predatorMask[x, y] = False
                continue
        neighbors = getNeighbors(x,y,predatorMask)
        if len(neighbors) < 7:
            moveRandom(x,y,predators, predatorMask)
        #- check for going off both bottom and right side of screen
        elif x + dx[i] >= m and y + dy[i] >= n:
            predators[0, 0, 0] = predators[x, y, 0] - PREDATOR_MOVE_ENERGY
            predators[0, 0, 1] = predators[x, y, 1]
            predatorMask[0, 0] = True
        #- check for going off right side of screen
        elif x + dx[i] >= m:
            predators[0, y + dy[i], 0] = \
                predators[x, y, 0] - PREDATOR_MOVE_ENERGY
            predators[0, y + dy[i], 1] = predators[x, y, 1]
            predatorMask[0, y + dy[i]] = True    
        #- check for going off bottom side of screen
        elif y + dy[i] >= n:
            predators[x + dx[i], 0, 0] = \
                predators[x, y, 0] - PREDATOR_MOVE_ENERGY
            predators[x + dx[i], 0, 1] = predators[x, y, 1]
            predatorMask[x + dx[i], 0] = True
        #- movement within bounds of screen
        else:
            predators[predatorIndices[i,0] + dx[i], predatorIndices[i,1] + dy[i], 0] = \
                predators[predatorIndices[i,0], predatorIndices[i,1], 0] - PREDATOR_MOVE_ENERGY
            predators[predatorIndices[i,0] + dx[i], predatorIndices[i,1] + dy[i], 1] = predators[predatorIndices[i,0], predatorIndices[i,1], 1]
            predatorMask[predatorIndices[i,0] + dx[i], predatorIndices[i,1] + dy[i]] = True
                
        #- remove signs of predator in old location
        predators[predatorIndices[i,0], predatorIndices[i,1], 0] = \
            predators[predatorIndices[i,0], predatorIndices[i,1], 1] = 0
        predatorMask[predatorIndices[i,0], predatorIndices[i,1]] = False
        
def movePrey(prey, preyMask, predators, predatorMask, plants):
    #- Array of indices of predator, prey, and plant locations
    preyIndices = np.argwhere(preyMask)
    dx = []
    dy = []
    for i in range(len(preyIndices)):
        #- (x, y) coordinate of predator in predatorMask
        x, y = preyIndices[i,0], preyIndices[i,1]
        #- eyesightArray gives us an array with (x,y) as the center and a radius of predator eyesight
        #- eyesight will contain any indices of prey within the eyesightArray
        plantEyesight = np.transpose(np.where(eyesightArray(plants, x, y, PREY_EYESIGHT*2+1) == 0))  
        predatorEyesight = np.argwhere(eyesightArray(predatorMask,x,y,PREY_EYESIGHT*2+1))  
        #- center of eyesightArray is located at (eyesight, eyesight)
        center = (PREY_EYESIGHT, PREY_EYESIGHT)
        #- Calculates relative distance to all predator from x,y coordinate of prey
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            predDistances = np.floor(np.sqrt(np.sum((predatorEyesight-center)**2, axis=1)))
            plantDistances = np.floor(np.sqrt(np.sum((plantEyesight-center)**2, axis=1)))
        else:
            #- Manhattan distance
            predDistances = np.sum(np.abs(predatorEyesight-center), axis=1)
            plantDistances = np.sum(np.abs(plantEyesight-center), axis=1)

        if len(predDistances) == 0:
            if len(plantDistances) == 0:
                dx.append(np.random.choice([-1,0,1]))
                dy.append(np.random.choice([-1,0,1]))
            else:
                closestPlantIndex = np.where(plantDistances == plantDistances.min())
                if len(closestPlantIndex[0]) > 1: closestPlantIndex = np.random.choice(closestPlantIndex[0])
                else: closestPlantIndex = closestPlantIndex[0][0]
                if plantEyesight[closestPlantIndex,0] > center[0]: dx.append(1)
                elif plantEyesight[closestPlantIndex,0] < center[0]: dx.append(-1)
                else: dx.append(0)
                if plantEyesight[closestPlantIndex,1] > center[1]: dy.append(1)
                elif plantEyesight[closestPlantIndex,1] < center[1]: dy.append(-1)
                else: dy.append(1)
        else:
            closestPredatorIndex = np.argmin(predDistances)
            if predatorEyesight[closestPredatorIndex,0] > center[0]: dx.append(-1)
            elif predatorEyesight[closestPredatorIndex,0] < center[0]: dx.append(1)
            else: dx.append(0)
            if predatorEyesight[closestPredatorIndex,1] > center[1]: dy.append(-1)
            elif predatorEyesight[closestPredatorIndex,1] < center[1]: dy.append(1)
            else: dy.append(1)
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    
    if not DIAGONAL_MOVEMENT:
        #- prioritize moving in x direction first (move in y once x coordinates line up)
        dy = np.where(dx == 0, dy, 0)
    
    #- movement validation for boundary checks and process movement in arrays
    for i in range(len(preyIndices)):
        x, y = preyIndices[i,0], preyIndices[i,1]
        #- if moving causes energy to equal or go below 0, prey dies
        if (prey[x, y, 0] - PREY_MOVE_ENERGY <= 0):
                prey[x, y, 0] = 0
                preyMask[x, y] = False
                continue
        neighbors = getNeighbors(x,y,preyMask)
        if len(neighbors) < 7:
            moveRandom(x,y,prey, preyMask)
        #- check for going off both bottom and right side of screen
        elif x + dx[i] >= m and y + dy[i] >= n:
            prey[0, 0, 0] = prey[x, y, 0] - PREY_MOVE_ENERGY
            prey[0, 0, 1] = prey[x, y, 1]
            preyMask[0, 0] = True
        #- check for going off right side of screen
        elif x + dx[i] >= m:
            prey[0, y + dy[i], 0] = \
                prey[x, y, 0] - PREY_MOVE_ENERGY
            prey[0, y + dy[i], 1] = prey[x, y, 1]
            preyMask[0, y + dy[i]] = True    
        #- check for going off bottom side of screen
        elif y + dy[i] >= n:
            prey[x + dx[i], 0, 0] = \
                prey[x, y, 0] - PREY_MOVE_ENERGY
            prey[x + dx[i], 0, 1] = prey[x, y, 1]
            preyMask[x + dx[i], 0] = True
        #- movement within bounds of screen
        else:
            prey[x + dx[i], y + dy[i], 0] = \
                prey[x, y, 0] - PREY_MOVE_ENERGY
            prey[x + dx[i], y + dy[i], 1] = prey[x, y, 1]
            preyMask[x + dx[i], y + dy[i]] = True
            
        #- remove signs of prey in old location
        prey[preyIndices[i,0], preyIndices[i,1], 0] = 0
        preyMask[preyIndices[i,0], preyIndices[i,1]] = False
        
def moveRandom(x, y, data, mask):
    neighbors = getNeighbors(x, y)
    randomNeighbor = neighbors[np.random.choice(len(neighbors))]
    data[randomNeighbor, 0] = \
        data[x, y, 0] - PREY_MOVE_ENERGY
    data[randomNeighbor, 1] = data[x, y, 1]
    mask[randomNeighbor] = True
        
def eyesightArray(a, x, y, n):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    a=np.roll(np.roll(a,shift=-x+int(n/2),axis=0),shift=-y+int(n/2),axis=1)
    return a[:n,:n]

def reproduce(prey, preyMask, predators, predatorMask):
    #- Array of indices of predator locations
    predatorIndices = np.argwhere(predatorMask)  
    for i in range(len(predatorIndices)):
        x, y = predatorIndices[i,0], predatorIndices[i,1]
        eyesight = np.argwhere(eyesightArray(predatorMask, x, y, REPRODUCTION_RANGE*2+1))  
        center = (REPRODUCTION_RANGE, REPRODUCTION_RANGE)
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
        else:
            #- Manhattan distance
            distances = np.sum(np.abs(eyesight-center), axis=1)
        if predators[x,y,0] < REPRODUCTION_THRESHOLD or predators[x,y,1] > 0:
            predators[x,y,1] -= 1
            continue
        else:
            #- get list of neighbor cells (x, y) coordinates to current predator
            neighbors = getNeighbors(x, y, predatorMask)
            newx, newy = neighbors[np.random.choice(len(neighbors))]
            predatorMask[newx, newy] = True
            predators[newx, newy, 0] = PREDATOR_START_ENERGY
            predators[newx, newy, 1] = PREDATOR_REPRODUCTION_START_TIME
            predators[x,y,1] = PREDATOR_REPRODUCTION_START_TIME
                
    preyIndices = np.argwhere(preyMask)
    for i in range(len(preyIndices)):
        x, y = preyIndices[i,0], preyIndices[i,1]
        eyesight = np.argwhere(eyesightArray(preyMask, x, y, REPRODUCTION_RANGE*2+1))  
        center = (REPRODUCTION_RANGE, REPRODUCTION_RANGE)
        if DIAGONAL_MOVEMENT:
            #- Euclidean distance
            distances = np.sqrt(np.sum((eyesight-center)**2, axis=1))
        else:
            #- Manhattan distance
            distances = np.sum(np.abs(eyesight-center), axis=1)
        if prey[x,y,0] < REPRODUCTION_THRESHOLD or prey[x,y,1] > 0:
            prey[x,y,1] -= 1
            continue
        else:
            #- get list of neighbor cells (x, y) coordinates to current prey
            neighbors = getNeighbors(x, y, preyMask)
            if len(neighbors) > 0:
                newx, newy = neighbors[np.random.choice(np.arange(len(neighbors)))]
                preyMask[newx, newy] = True
                prey[newx, newy, 0] = PREDATOR_START_ENERGY
                prey[newx, newy, 1] = PREY_REPRODUCTION_START_TIME
                prey[x,y,1] = PREY_REPRODUCTION_START_TIME

def getNeighbors(x, y, mask=None):
    #- get list of neighbor cells (x, y) coordinates to current predator
    neighbors = [(x-1, y-1),(x, y-1),(x+1, y-1),(x-1, y),(x+1, y),(x-1, y+1),(x, y+1),(x+1, y+1)]
    for i in range(len(neighbors)):
        #- check for going off both bottom and right side of screen
        if neighbors[i][0] == m and neighbors[i][1] == n:
            neighbors[i] = (0, 0)
        #- check for going off right side of screen
        elif neighbors[i][0] == m:
            neighbors[i] = (0, neighbors[i][1])
        #- check for going off bottom side of screen
        elif neighbors[i][1] == n:
            neighbors[i] = (neighbors[i][0], 0)
    if mask is not None:
        for neighbor in neighbors:
            if mask[neighbor[0],neighbor[1]]:
                neighbors.remove(neighbor)
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

