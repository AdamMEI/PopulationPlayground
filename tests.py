#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:38:22 2023

@author: adamthepig
"""

import population_playground
import numpy as np

def spawnTest():
    prey, preyMask, predator, predatorMask, plants, plantMask = \
        population_playground.initialize()
    population_playground.visualize(preyMask, predatorMask, plantMask)

def testFeed(y, x):
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        population_playground.initialize()
    prey[y, x, 0] = population_playground.PREY_START_ENERGY
    prey[y, x, 1] = population_playground.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    predators[y, x, 0] = 15
    predators[y, x, 1] = population_playground.PREDATOR_REPRODUCTION_START_TIME
    predatorMask[y, x] = True
    population_playground.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    return (prey, preyMask, predators, predatorMask, plants, plantMask)

def stunTest():
    y = 4
    x = 4
    population_playground.STUN_CHANCE = 1.0
    population_playground.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(preyMask[y, x])
    assert(predatorMask[y, x])
    assert(predators[y, x, 2] == population_playground.STUN_TIME)

def killTest():
    y = 4
    x = 4
    population_playground.STUN_CHANCE = 0.0
    population_playground.PREY_KILL_CHANCE = 1.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(preyMask[y, x])
    assert(not predatorMask[y, x])
    assert(np.array_equal(predators[y, x], np.array([0, 0, 0])))

def eatTest():
    y = 4
    x = 4
    population_playground.STUN_CHANCE = 0.0
    population_playground.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(not preyMask[y, x])
    assert(predatorMask[y, x])
    assert(np.array_equal(prey[y, x], np.array([0, 0])))
    assert(predators[y, x, 0] == min( \
        15 + population_playground.PREDATOR_EAT_ENERGY,
        population_playground.PREDATOR_MAX_ENERGY))

def eatPlantTest():
    y = 4
    x = 4
    population_playground.STUN_CHANCE = 0.0
    population_playground.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        population_playground.initialize()
    prey[y, x, 0] = 15
    prey[y, x, 1] = population_playground.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    plantMask[y, x] = True
    population_playground.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    assert(preyMask[y, x])
    assert(plantMask[y, x])
    assert(plants[y, x] == population_playground.PLANT_REGROWTH_TIME)
    assert(prey[y, x, 0] == min( \
        15 + population_playground.PREY_EAT_ENERGY,
        population_playground.PREY_MAX_ENERGY))

def eatUngrownPlantTest():
    y = 4
    x = 4
    population_playground.STUN_CHANCE = 0.0
    population_playground.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        population_playground.initialize()
    prey[y, x, 0] = 15
    prey[y, x, 1] = population_playground.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    plantMask[y, x] = True
    plants[y, x] = population_playground.PLANT_REGROWTH_TIME
    population_playground.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    assert(preyMask[y, x])
    assert(plantMask[y, x])
    assert(plants[y, x] == population_playground.PLANT_REGROWTH_TIME - 1)
    assert(prey[y, x, 0] == 15)
    
    

stunTest()
killTest()
eatTest()
eatPlantTest()
eatUngrownPlantTest()