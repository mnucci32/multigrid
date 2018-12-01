#!/usr/bin/python3

import multigrid as mg
import simulationData as sd
import numpy as np
import time
import optparse

def main():
  # Set up options
  parser = optparse.OptionParser()
  parser.add_option("-x", "--xSpan", action="store", dest="xSpan",
                    default="0.0 1.0 11",
                    help="x-coordinate span and length. Default = 0 1 11")
  parser.add_option("-y", "--ySpan", action="store", dest="ySpan",
                    default="0.0 1.0 11",
                    help="y-coordinate span and length. Default = 0 1 11")
  parser.add_option("-v", "--nu", action="store", dest="nu",
                    default="1.0",
                    help="nu coefficient for heat equation. Default = 1.0")
  parser.add_option("-n", "--timeSteps", action="store", dest="timeSteps",
                    default="1.0",
                    help="number of time steps. Default = 1")
  parser.add_option("-t", "--threshold", action="store", dest="threshold",
                    default="1.0e-10",
                    help="residual threshold to use for timing. " \
                    + "Default = 1e-10")
  parser.add_option("-c", "--cornerTemperatures", action="store",
                    dest="cornerTemps",
                    default="100 150 150 200",
                    help="corner temperatures to use as BCs. " \
                    + "Format is (xl,yl xu,yl yu,xl, xu,yu). " \
                    + "Default = 100 150 150 200")

  options, remainder = parser.parse_args()
  simData = sd.simulationData(options, 3, "V")

  t0 = time.time()

  # construct grids
  solution = mg.mgSolution(simData)

  # march solution in time
  print("Iteration          Residual          Time")
  for nn in simData.iteration:
    solution.MultigridCycle()
    solution.ResidNorm(nn, t0)

  # print and plot solution
  solution.PlotNode()
  solution.Print()

if __name__ == "__main__":
  main()
