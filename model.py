#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:58:49 2023

@author: adamthepig
"""

import numpy as np
import math
import random
import pygame

m = 80
n = 80
PREY_START_NUM = 30
PREDATOR_START_NUM = 10
PLANT_START_NUM = 10
PREY_START_ENERGY = 100
PREDATOR_START_ENERGY = 100
PREY_EAT_ENERGY = 15
PREDATOR_EAT_ENERGY = 50
PREY_MAX_ENERGY = 100
PREDATOR_MAX_ENERGY = 100
PREY_REPRODUCTION_START_TIME = 100
PREDATOR_REPRODUCTION_START_TIME = 100
PLANT_REGROWTH_TIME = 100
STUN_CHANCE = 0.1
PREY_KILL_CHANCE = 0.1
STUN_TIME = 5
PIXEL_SIZE = 10
BACKGROUND_COLOR = (255, 255, 255)
PREY_COLOR = (0, 0, 255)
PREDATOR_COLOR = (255, 0, 0)
PLANT_GROWN_COLOR = (0, 255, 0)
PLANT_UNGROWN_COLOR = (255, 255, 0)

def initialize():
    prey = np.zeros((m, n, 2))
    preyMask = np.zeros((m, n), dtype=bool)
    i = 0
    while i < PREY_START_NUM:
        x = random.randint(1, n - 2)
        y = random.randint(1, m - 2)
        if not preyMask[y, x]:
            preyMask[y, x] = True
            i += 1
    prey[preyMask, 0] = PREY_START_ENERGY
    prey[preyMask, 1] = PREY_REPRODUCTION_START_TIME
    predators = np.zeros((m, n, 3))
    predatorMask = np.zeros((m, n), dtype=bool)
    i = 0
    while i < PREDATOR_START_NUM:
        x = random.randint(1, n - 2)
        y = random.randint(1, m - 2)
        if not predatorMask[y, x] and not preyMask[y, x]:
            predatorMask[y, x] = True
            i += 1
    predators[predatorMask, 0] = PREDATOR_START_ENERGY
    predators[predatorMask, 1] = PREDATOR_REPRODUCTION_START_TIME
    plantMask = np.zeros((m, n), dtype=bool)
    plantCenterMask = np.zeros((m, n), dtype=bool)
    i = 0
    while i < PLANT_START_NUM:
        x = random.randint(2, n - 3)
        y = random.randint(2, m - 3)
        if not plantCenterMask[y, x]:
            plantCenterMask[y, x] = True
            i += 1
    plantMask[1:-1, 1:-1] = plantCenterMask[0:-2,0:-2] + \
        plantCenterMask[0:-2,1:-1] + plantCenterMask[0:-2,2:] + \
        plantCenterMask[1:-1,0:-2] + plantCenterMask[1:-1,1:-1] + \
        plantCenterMask[1:-1,2:] + plantCenterMask[2:,0:-2] + \
        plantCenterMask[2:,1:-1] + plantCenterMask[2:,2:]
    plants = np.zeros((m, n))
    return (prey, preyMask, predators, predatorMask, \
            plants, plantMask)

def visualize(preyMask, predatorMask, plantMask, plants):
    pygame.init()
    screen = pygame.display.set_mode([m * PIXEL_SIZE, n * PIXEL_SIZE])
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((255, 255, 255))
        for i in range(m):
            for j in range(n):
                rect = pygame.Rect(i * PIXEL_SIZE, j * PIXEL_SIZE,
                                   PIXEL_SIZE, PIXEL_SIZE)
                if preyMask[i, j]:
                    pygame.draw.rect(screen, PREY_COLOR, rect)
                elif predatorMask[i, j]:
                    pygame.draw.rect(screen, PREDATOR_COLOR, rect)
                elif plantMask[i, j]:
                    plantUngrownColor = np.array(PLANT_UNGROWN_COLOR)
                    plantGrownColor = np.array(PLANT_GROWN_COLOR)
                    plantColor = plantUngrownColor * plants[i, j] / \
                        PLANT_REGROWTH_TIME + plantGrownColor * (1 -
                        (plants[i, j] / PLANT_REGROWTH_TIME))
                    pygame.draw.rect(screen, plantColor, rect)
        pygame.display.flip()
    pygame.quit()

def feed(prey, preyMask, predators, predatorMask, plants, plantMask):
    overlappingMask = preyMask * predatorMask
    stunnedChances = np.random.random((m, n))
    stunnedMask = (stunnedChances < STUN_CHANCE) * overlappingMask
    predators[stunnedMask, 2] = STUN_TIME
    
    killChances = np.random.random((m, n))
    killMask = (killChances < PREY_KILL_CHANCE) * overlappingMask
    predators[killMask] = 0
    predatorMask[killMask] = False
    eatMask = (overlappingMask * np.logical_not( \
        np.logical_or(stunnedMask, killMask)))
    prey[eatMask] = 0
    preyMask[eatMask] = False
    predators[eatMask, 0] = np.where( \
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY < PREDATOR_MAX_ENERGY,
        predators[eatMask, 0] + PREDATOR_EAT_ENERGY, PREDATOR_MAX_ENERGY)
    
    growingPlantMask = plants > 0
    plants[growingPlantMask] -= 1
    overlappingPlantMask = preyMask * plantMask
    grownOverlappingPlantMask = overlappingPlantMask * \
                                np.logical_not(growingPlantMask)
    plants[grownOverlappingPlantMask] = PLANT_REGROWTH_TIME
    prey[grownOverlappingPlantMask, 0] = np.where( \
        prey[grownOverlappingPlantMask, 0] + PREY_EAT_ENERGY < PREY_MAX_ENERGY,
        prey[grownOverlappingPlantMask, 0] + PREY_EAT_ENERGY, PREY_MAX_ENERGY)
    

if (__name__ == '__main__'):
    prey, preyMask, predators, predatorMask, plants, plantMask = initialize()
    feed(prey, preyMask, predators, predatorMask, plants, plantMask)
    visualize(preyMask, predatorMask, plantMask, plants)