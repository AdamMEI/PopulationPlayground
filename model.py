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

#--------------------------------- CONSTANTS ----------------------------------

#- Number of sets of simulations run
SET_NUM = 10
#- Number of simulations run per set
SIMULATION_NUM = 10
#- Number of time steps in the simulation
TIME_STEPS = 1000
#- Grid size
m = 80 #- Vertical
n = 80 #- Horizontal
#- The starting number of prey, predators, and plants
PREY_START_NUM = 30
PREDATOR_START_NUM = 10
PLANT_START_NUM = 10
#- The energy that prey and predators start with
PREY_START_ENERGY = 100
PREDATOR_START_ENERGY = 100
#- The amount of energy gained by prey and predators after eating
PREY_EAT_ENERGY = 15
PREDATOR_EAT_ENERGY = 50
#- The maximum energy that prey and predators can have
PREY_MAX_ENERGY = 100
PREDATOR_MAX_ENERGY = 100
#- The amount of time until prey and predators can reproduce for the first time
PREY_REPRODUCTION_START_TIME = 100
PREDATOR_REPRODUCTION_START_TIME = 100
#- The time required for plants to regrow after being eaten
PLANT_REGROWTH_TIME = 100
#- The chance of prey fighting back and stunning or killing predators
STUN_CHANCE = 0.1
PREY_KILL_CHANCE = 0.1
#- The amount of time that predators are stunned for after prey stun them
STUN_TIME = 5
#- The number of pixels for each grid location
PIXEL_SIZE = 10
#- The visualization colors
BACKGROUND_COLOR = (255, 255, 255)
PREY_COLOR = (0, 0, 255)
PREDATOR_COLOR = (255, 0, 0)
#- Plant colors are interpolated between these two
PLANT_GROWN_COLOR = (0, 255, 0)
PLANT_UNGROWN_COLOR = (255, 255, 0)

def runSets():
    """
    Runs multiple sets of simulations, then prints average populations at the
    end of the simulation and the standard deviation of the populations.
    """
    preyPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    predatorPopulations = np.zeros((SET_NUM, SIMULATION_NUM, TIME_STEPS))
    for i in range(SET_NUM):
        preyPopulations[i], predatorPopulations[i] = runSet()
    print(
f"""
Populations at end of simulation
Averages
    Prey: {np.average(preyPopulations[:, :, -1])}
    Predators: {np.average(predatorPopulations[:, :, -1])}
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
    preyPopulations = np.zeros(TIME_STEPS)
    predatorPopulations = np.zeros(TIME_STEPS)
    #- Initializes the program
    prey, preyMask, predators, predatorMask, plants, plantMask = initialize()
    if shouldVisualize:
        screen = initVisualization()
    for i in range(TIME_STEPS):
        #- Runs a single feed cycle
        feed(prey, preyMask, predators, predatorMask, plants, plantMask)
        if shouldVisualize:
            #- Visualizes the grid
            if not visualize(screen, preyMask, predatorMask,
                             plantMask, plants):
                break
        preyPopulations[i] = np.count_nonzero(preyMask)
        predatorPopulations[i] = np.count_nonzero(predatorMask)
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
    plantMask : 2d boolean array
        The locations that contain plants
    """
    #- Create arrays
    prey = np.zeros((m, n, 2))
    preyMask = np.zeros((m, n), dtype=bool)
    predators = np.zeros((m, n, 3))
    predatorMask = np.zeros((m, n), dtype=bool)
    plantMask = np.zeros((m, n), dtype=bool)
    plantCenterMask = np.zeros((m, n), dtype=bool)
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
    i = 0
    while i < PLANT_START_NUM:
        x = random.randint(2, n - 3)
        y = random.randint(2, m - 3)
        #- Ensures that there is not already a plant there (plants can still
        #  overlap slightly because they are 3x3)
        if not plantCenterMask[y, x]:
            plantCenterMask[y, x] = True
            i += 1
    #- Sets prey and predator start values
    prey[preyMask, 0] = PREY_START_ENERGY
    prey[preyMask, 1] = PREY_REPRODUCTION_START_TIME
    predators[predatorMask, 0] = PREDATOR_START_ENERGY
    predators[predatorMask, 1] = PREDATOR_REPRODUCTION_START_TIME
    #- Adds other plants around the centers of the plants to make them 3x3
    plantMask[1:-1, 1:-1] = np.array(plantCenterMask[0:-2,0:-2] + \
        plantCenterMask[0:-2,1:-1] + plantCenterMask[0:-2,2:] + \
        plantCenterMask[1:-1,0:-2] + plantCenterMask[1:-1,1:-1] + \
        plantCenterMask[1:-1,2:] + plantCenterMask[2:,0:-2] + \
        plantCenterMask[2:,1:-1] + plantCenterMask[2:,2:], bool)
    #- Initializes plants
    plants = plantMask * PLANT_REGROWTH_TIME
    return (prey, preyMask, predators, predatorMask, \
            plants, plantMask)

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

def visualize(screen, preyMask, predatorMask, plantMask, plants):
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
    plantMask : 2d boolean array
        The locations that contain plants
    """
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
            elif plantMask[i, j]:
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
    return True

def feed(prey, preyMask, predators, predatorMask, plants, plantMask):
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
    plantMask : 2d boolean array
        The locations that contain plants
    """
    #- Mask of overlapping prey and predators
    overlappingMask = preyMask * predatorMask
    #- Stuns predators
    stunnedChances = np.random.random((m, n))
    stunnedMask = (stunnedChances < STUN_CHANCE) * overlappingMask
    predators[stunnedMask, 2] = STUN_TIME
    #- Kills predators (predators both stunned and killed will die)
    killChances = np.random.random((m, n))
    killMask = (killChances < PREY_KILL_CHANCE) * overlappingMask
    predators[killMask] = 0
    predatorMask[killMask] = False
    #- Eats prey (if the predator was not stunned or killed)
    eatMask = (overlappingMask * np.logical_not( \
        np.logical_or(stunnedMask, killMask)))
    prey[eatMask] = 0
    preyMask[eatMask] = False
    #- Gives predators that ate energy
    predators[eatMask, 0] = np.where( \
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY < PREDATOR_MAX_ENERGY,
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY, PREDATOR_MAX_ENERGY)
    #- Grows plants that are not fully grown
    growingPlantMask = plants > 0
    plants[growingPlantMask] -= 1
    #- Checks which plants overlap with prey and are fully grown
    overlappingPlantMask = preyMask * plantMask
    grownOverlappingPlantMask = overlappingPlantMask * \
                                np.logical_not(growingPlantMask)
    #- Resets eaten plants and gives prey energy from eating the plants
    plants[grownOverlappingPlantMask] = PLANT_REGROWTH_TIME
    prey[grownOverlappingPlantMask, 0] = np.where( \
        prey[grownOverlappingPlantMask, 0] + PREY_EAT_ENERGY < PREY_MAX_ENERGY,
        prey[grownOverlappingPlantMask, 0] + PREY_EAT_ENERGY, PREY_MAX_ENERGY)
    
#- Checks if this file is being run directly and not imported
if (__name__ == '__main__'):
    runSimulation(True)