#!/usr/bin/python3

import multigrid as mg
import numpy as np
import time

def main():
  # set up inputs
  xRange = [0.0, 10.0]
  xNum = 21
  yRange = [0.0, 10.0]
  yNum = 21
  gridLevels = 2
  nu = 1
  timeSteps = 100

  t0 = time.time()

  # construct grids
  solution = mg.mgSolution(xRange, xNum, yRange, yNum, nu, gridLevels)

  # march solution in time
  print("Iteration          Residual          Time")
  for nn in range(0, timeSteps):
    solution.MultigridCycle()
    solution.ResidNorm(nn, t0)

  # print and plot solution
  #solution.Print()
  solution.PlotNode()

if __name__ == "__main__":
  main()
