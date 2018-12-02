#!/usr/bin/python3

import multigrid as mg
import simulationData as sd
import numpy as np
import time
import optparse
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 20


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

  # ------------------------------------------------------------------
  # baseline simulation - no multigrid
  print("-----------------------------------------")
  print("Baseline Simulation")
  print("-----------------------------------------")
  baselineData = sd.simulationData(options, 1, "V", "Baseline")
  # construct grids
  baseline = mg.mgSolution(baselineData)
  # march solution in time
  print("Iteration          Residual          Time")
  for nn in baselineData.iteration:
    baseline.MultigridCycle()
    resid = baseline.ResidNorm(nn, baselineData.startingTime)
    baselineData.LogResidual(nn, resid)
  print("\n\n")

  # ------------------------------------------------------------------
  # 2 level multigrid, V cycle
  print("-----------------------------------------")
  print("2 Level Multigrid V Cycle")
  print("-----------------------------------------")
  mg2vData = sd.simulationData(options, 2, "V", "2 Level V")
  # construct grids
  mg2v = mg.mgSolution(mg2vData)
  # march solution in time
  print("Iteration          Residual          Time")
  for nn in mg2vData.iteration:
    mg2v.MultigridCycle()
    resid = mg2v.ResidNorm(nn, mg2vData.startingTime)
    mg2vData.LogResidual(nn, resid)
  print("\n\n")

  # ------------------------------------------------------------------
  # 4 level multigrid, V cycle
  print("-----------------------------------------")
  print("4 Level Multigrid V Cycle")
  print("-----------------------------------------")
  mg4vData = sd.simulationData(options, 4, "V", "4 Level V")
  # construct grids
  mg4v = mg.mgSolution(mg4vData)
  # march solution in time
  print("Iteration          Residual          Time")
  for nn in mg4vData.iteration:
    mg4v.MultigridCycle()
    resid = mg4v.ResidNorm(nn, mg4vData.startingTime)
    mg4vData.LogResidual(nn, resid)
  print("\n\n")

  # ------------------------------------------------------------------
  # 4 level multigrid, W cycle
  print("-----------------------------------------")
  print("4 Level Multigrid W Cycle")
  print("-----------------------------------------")
  mg4wData = sd.simulationData(options, 4, "W", "4 Level W")
  # construct grids
  mg4w = mg.mgSolution(mg4wData)
  # march solution in time
  print("Iteration          Residual          Time")
  for nn in mg4wData.iteration:
    mg4w.MultigridCycle()
    resid = mg4w.ResidNorm(nn, mg4wData.startingTime)
    mg4wData.LogResidual(nn, resid)
  print("\n\n")

  # ------------------------------------------------------------------
  # plot solutions
  _, ax = plt.subplots(2, 3, figsize=(24, 12))
  baseline.PlotNode(ax[0, 0], baselineData.name)
  mg2v.PlotNode(ax[0, 1], mg2vData.name)
  mg4v.PlotNode(ax[0, 2], mg4vData.name)
  mg4w.PlotNode(ax[1, 0], mg4wData.name)
  # plot residuals
  ax[1, 2].set_xlabel("Iteration")
  ax[1, 2].set_ylabel("Residual")
  ax[1, 2].set_title("Residuals")
  ax[1, 2].semilogy(baselineData.iteration, baselineData.residuals, "k", lw=3)
  ax[1, 2].semilogy(mg2vData.iteration, mg2vData.residuals, "b", lw=3)
  ax[1, 2].semilogy(mg4vData.iteration, mg4vData.residuals, "r", lw=3)
  ax[1, 2].semilogy(mg4wData.iteration, mg4wData.residuals, "g", lw=3)
  ax[1, 2].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name])
  ax[1, 2].grid(True)
  plt.tight_layout()
  plt.show()

  print("-----------------------------------------")
  print("Summary")
  print("-----------------------------------------")
  baselineData.PrintTimeToThreshold()
  mg2vData.PrintTimeToThreshold()
  mg4vData.PrintTimeToThreshold()
  mg4wData.PrintTimeToThreshold()


if __name__ == "__main__":
  main()
