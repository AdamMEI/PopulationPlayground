#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs a variety of tests on the model.py file.
"""

#------------------------------------------------------------------------------
#                           Additional Documentation
#
# Modification History:
# - Viewable on GitHub: https://github.com/AdamMEI/PopulationPlayground
#
# Authors
# - Gabriel Schepman, Applied Computing, University of Washington Bothel
# - Adam Meilicke, Computer Science and Softare Engineering, University of
#   Washington Bothell
#
# Notes:
# - Written for Python 3.10.
# - Model can be tested through visualization and plot analysis. To test, type
#   "python model.py".
#
#==============================================================================

#------------------------------- MODULE IMPORTS -------------------------------

import model
import numpy as np

#----------------------------------- TESTS ------------------------------------

def spawnTest():
    """
    Tests random prey, plant, and predator spawning.
    """
    prey, preyMask, predator, predatorMask, plantMask = \
        model.initialize()
    screen = model.initVisualization()
    running = True
    while running:
        model.visualize(screen, preyMask, predator, predatorMask, plantMask)
        for event in model.pygame.event.get():
            if event.type == model.pygame.QUIT:
                running = False
    model.pygame.quit()

def testFeed(y, x):
    """
    Util function used by other testing functions to avoid rewriting code.

    Parameters
    ----------
    y : int
        y coordinate to test
    x : int
        x coordinate to test

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
    prey, preyMask, predators, predatorMask, plants = \
        model.initialize()
    prey[y, x, 0] = model.PREY_START_ENERGY
    prey[y, x, 1] = model.PREY_REPRODUCTION_TIME[0]
    preyMask[y, x] = True
    predators[y, x, 0] = 15
    predators[y, x, 1] = model.PREDATOR_REPRODUCTION_TIME[0]
    predatorMask[y, x] = True
    model.feed(prey, preyMask, predators, predatorMask,
                               plants)
    return (prey, preyMask, predators, predatorMask, plants)

def stunTest():
    """
    Tests stunning by changing the STUN_CHANCE to 1 (100%) and placing a prey
    and predator on top of each other.
    """
    y = 4
    x = 4
    model.STUN_CHANCE = 1.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants = testFeed(y, x)
    assert(preyMask[y, x])
    assert(predatorMask[y, x])
    assert(predators[y, x, 2] == model.STUN_TIME)

def killTest():
    """
    Tests defending by changing the PREY_KILL_CHANCE to 1 (100%) and placing a
    prey and predator on top of each other.
    """
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 1.0
    prey, preyMask, predators, predatorMask, plants = testFeed(y, x)
    assert(preyMask[y, x])
    assert(not predatorMask[y, x])
    #assert(np.array_equal(predators[y, x], np.array([0, 0, 0])))

def eatTest():
    """
    Tests eating by changing the PREY_KILL_CHANCE and STUN_CHANCE to 0 (0%)
    and placing a prey and predator on top of each other.
    """
    y = 4
    x = 4
    model.STUN_CHANCE = 0.0
    model.PREY_KILL_CHANCE = 0.0
    prey, preyMask, predators, predatorMask, plants = testFeed(y, x)
    assert(not preyMask[y, x])
    assert(predatorMask[y, x])
    assert(np.array_equal(prey[y, x], np.array([0, 0])))
    assert(predators[y, x, 0] == min( \
        15 + model.PREDATOR_EAT_ENERGY,
        model.PREDATOR_MAX_ENERGY))

def eatPlantTest():
    """
    Tests eating plants by placing a prey and grown plant on top of each other.
    """
    y = 4
    x = 4
    prey, preyMask, predators, predatorMask, plants = \
        model.initialize()
    prey[y, x, 0] = 3
    prey[y, x, 1] = model.PREY_REPRODUCTION_TIME[0]
    preyMask[y, x] = True
    model.feed(prey, preyMask, predators, predatorMask,
                               plants)
    assert(preyMask[y, x])
    assert(plants[y, x] == model.PLANT_REGROWTH_TIME)
    assert(prey[y, x, 0] == min( \
        3 + model.PREY_EAT_ENERGY,
        model.PREY_MAX_ENERGY))

def eatUngrownPlantTest():
    """
    Tests eating (or in this case, not eating) ungrown plants by placing a prey
    and ungrown plant on top of each other.
    """
    y = 4
    x = 4
    prey, preyMask, predators, predatorMask, plants = \
        model.initialize()
    prey[y, x, 0] = 15
    prey[y, x, 1] = model.PREY_REPRODUCTION_TIME[0]
    preyMask[y, x] = True
    plants[y, x] = model.PLANT_REGROWTH_TIME
    model.feed(prey, preyMask, predators, predatorMask,
                               plants)
    assert(preyMask[y, x])
    assert(plants[y, x] == model.PLANT_REGROWTH_TIME - 1)
    assert(prey[y, x, 0] == 15)

spawnTest()
print("PASSED - Spawn Test")
stunTest()
print("PASSED - Stun Test")
killTest()
print("PASSED - Kill Test")
eatTest()
print("PASSED - Eat Test")
eatPlantTest()
print("PASSED - Eat Plant Test")
eatUngrownPlantTest()
print("PASSED - Eat Ungrown Plant Test")