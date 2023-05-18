#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:38:22 2023

@author: adamthepig
"""

import model
import numpy as np

def spawnTest():
    """
    Tests the random spawning of prey, predators, and plants.
    """
    prey, preyMask, predator, predatorMask, plants, plantMask = \
        model.initialize()
    model.visualize(preyMask, predatorMask, plantMask, plants)

def testFeed(y, x):
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        model.initialize()
    prey[y, x, 0] = model.PREY_START_ENERGY
    prey[y, x, 1] = model.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    predators[y, x, 0] = 15
    predators[y, x, 1] = model.PREDATOR_REPRODUCTION_START_TIME
    predatorMask[y, x] = True
    model.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    return (prey, preyMask, predators, predatorMask, plants, plantMask)

def stunTest():
    y = 4
    x = 4
    model.STUN_CHANCE = 1.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(preyMask[y, x])
    assert(predatorMask[y, x])
    assert(predators[y, x, 2] == model.STUN_TIME)

def killTest():
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 1.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(preyMask[y, x])
    assert(not predatorMask[y, x])
    assert(np.array_equal(predators[y, x], np.array([0, 0, 0])))

def eatTest():
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = testFeed(y, x)
    assert(not preyMask[y, x])
    assert(predatorMask[y, x])
    assert(np.array_equal(prey[y, x], np.array([0, 0])))
    assert(predators[y, x, 0] == min( \
        15 + model.PREDATOR_EAT_ENERGY,
        model.PREDATOR_MAX_ENERGY))

def eatPlantTest():
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        model.initialize()
    prey[y, x, 0] = 15
    prey[y, x, 1] = model.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    plantMask[y, x] = True
    model.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    assert(preyMask[y, x])
    assert(plantMask[y, x])
    assert(plants[y, x] == model.PLANT_REGROWTH_TIME)
    assert(prey[y, x, 0] == min( \
        15 + model.PREY_EAT_ENERGY,
        model.PREY_MAX_ENERGY))

def eatUngrownPlantTest():
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants, plantMask = \
        model.initialize()
    prey[y, x, 0] = 15
    prey[y, x, 1] = model.PREY_REPRODUCTION_START_TIME
    preyMask[y, x] = True
    plantMask[y, x] = True
    plants[y, x] = model.PLANT_REGROWTH_TIME
    model.feed(prey, preyMask, predators, predatorMask,
                               plants, plantMask)
    assert(preyMask[y, x])
    assert(plantMask[y, x])
    assert(plants[y, x] == model.PLANT_REGROWTH_TIME - 1)
    assert(prey[y, x, 0] == 15)
    
    
spawnTest()
stunTest()
killTest()
eatTest()
eatPlantTest()
eatUngrownPlantTest()