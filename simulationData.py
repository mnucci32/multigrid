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
    self.maxTemp = float(options.maxTemp)
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
    # sanity checks
    self.SanityCheck()

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

  def SanityCheck(self):
    nx = len(self.xc)
    ny = len(self.yc)
    gridSizes = np.zeros((4, 2), dtype=int)
    gridSizes[0, :] = [nx, ny]
    # check that 3 additional grid levels are possible
    for ii in range(1, 4):
      if (nx - 1) % 2 != 0:
        print("ERROR in grid refinement", ii, "for x dimension.")
        print(nx, "- 1 is not divisible by 2")
        print("Grid Level  X-Dimension")
        for jj in range(0, ii):
          print("    ", jj, "        ", gridSizes[jj, 0])
        print("Please choose a dimension that can support 4 grid levels")
        exit()
      nx = (nx - 1) / 2 + 1
      if (ny - 1) % 2 != 0:
        print("ERROR in grid refinement", ii, "for y dimension.")
        print(ny, "- 1 is not divisible by 2")
        print("Grid Level  Y-Dimension")
        for jj in range(0, ii):
          print("    ", jj, "        ", gridSizes[jj, 1])
        print("Please choose a dimension that can support 4 grid levels")
        exit()
      ny = (ny - 1) / 2 + 1
      gridSizes[ii, :] = [nx, ny]


