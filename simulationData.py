#!/usr/bin/python3

import numpy as np
import time

class simulationData:

  def __init__(self, options, gridLevels, cycle, name):
    # data common to all simulations
    self.nu = float(options.nu)
    tokens = options.xSpan.split()
    self.xc = np.linspace(float(tokens[0]), float(tokens[1]), int(tokens[2]))
    tokens = options.ySpan.split()
    self.yc = np.linspace(float(tokens[0]), float(tokens[1]), int(tokens[2]))
    self.timeSteps = int(options.timeSteps)
    self.cornerTemp = float(options.cornerTemp)
    self.iteration = np.zeros((self.timeSteps + 1))
    self.iteration = range(0, self.timeSteps + 1)
    self.residualThreshold = float(options.threshold)
    # data different for each simulation
    self.startingTime = time.time()
    self.timeToThreshold = -1.0
    self.gridLevels = gridLevels
    self.cycleType = cycle
    self.residuals = np.zeros((self.timeSteps + 1))
    self.times = np.zeros((self.timeSteps + 1))
    self.name = name

  def LogResidual(self, nn, l2, linf):
    self.residuals[nn] = l2
    nresid = l2 / self.residuals[0]
    dt = time.time() - self.startingTime
    self.times[nn] = dt
    print("{0:5d} {1:21.4e} {2:16.4e} {3:15.4e}".format(nn, nresid, linf, dt))
    if nresid <= self.residualThreshold and self.timeToThreshold < 0.0:
      self.timeToThreshold = time.time() - self.startingTime

  def NormResids(self):
    return self.residuals / self.residuals[0]

  def PrintTimeToThreshold(self):
    if self.timeToThreshold > 0:
      print(self.name, "reached threshold in",
            "{0:6.4e} s; final residual of {1:6.4e}".format(
                self.timeToThreshold, self.residuals[-1] / self.residuals[0]))
    else:
      print(self.name, "did not reach threshold")


