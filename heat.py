#!/usr/bin/python3

import multigrid as mg
import numpy as np

def main():
  # set up inputs
  xRange = [0.0, 3.0]
  xNum = 5
  yRange = [0.0, 3.0]
  yNum = 5
  gridLevels = 1
  nu = 1
  timeSteps = 10

  # construct grids
  solution = mg.mgSolution(xRange, xNum, yRange, yNum, nu, gridLevels)

  # march solution in time
  print("Iteration", "Residual")
  for nn in range(0, timeSteps):
    solution.levels[0].GaussSeidel()
    solution.levels[0].CalcResidual()
    solution.ResidNorm(nn)

  # print and plot solution
  solution.Print()
  solution.PlotNode()

if __name__ == "__main__":
  main()
