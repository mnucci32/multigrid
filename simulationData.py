#!/usr/bin/python3

import numpy as np
import time

class simulationData:

  def __init__(self, options, gridLevels, cycle):
    # data common to all simulations
    self.nu = float(options.nu)
    tokens = options.xSpan.split()
    self.xc = np.linspace(float(tokens[0]), float(tokens[1]), int(tokens[2]))
    tokens = options.ySpan.split()
    self.yc = np.linspace(float(tokens[0]), float(tokens[1]), int(tokens[2]))
    self.timeSteps = int(options.timeSteps)
    tokens = options.cornerTemps.split()
    self.cornerTemps = np.array([float(tokens[0]), float(tokens[1]), \
                                 float(tokens[2]), float(tokens[3])])
    self.iteration = np.zeros((self.timeSteps))
    self.iteration = range(0, self.timeSteps)
    self.residualThreshold = float(options.threshold)
    # data different for each simulation
    self.startingTime = time.time()
    self.timeToThreshold = -1.0
    self.gridLevels = gridLevels
    self.cycleType = cycle
    self.residuals = np.zeros((self.timeSteps))

  def LogResidual(self, nn, resid):
    self.residuals[nn] = resid
    if resid <= self.residualThreshold and self.timeToThreshold < 0.0:
      self.timeToThreshold = time.time() - self.startingTime


