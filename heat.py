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
  parser.add_option("-m", "--maxTemperature", action="store",
                    dest="maxTemp", default="500",
                    help="maximum temperature in domain. " \
                    + "Default = 500")

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
  baselineData.PrintHeaders()
  l2, linf = baseline.ResidNorm()
  baselineData.LogResidualAndError(0, l2, linf, baseline.ErrorNorm())
  for nn in range(1, baselineData.timeSteps + 1):
    baseline.MultigridCycle()
    l2, linf = baseline.ResidNorm()
    baselineData.LogResidualAndError(nn, l2, linf, baseline.ErrorNorm())
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
  mg2vData.PrintHeaders()
  l2, linf = mg2v.ResidNorm()
  mg2vData.LogResidualAndError(0, l2, linf, mg2v.ErrorNorm())
  for nn in range(1, mg2vData.timeSteps + 1):
    mg2v.MultigridCycle()
    l2, linf = mg2v.ResidNorm()
    mg2vData.LogResidualAndError(nn, l2, linf, mg2v.ErrorNorm())
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
  mg4vData.PrintHeaders()
  l2, linf = mg4v.ResidNorm()
  mg4vData.LogResidualAndError(0, l2, linf, mg4v.ErrorNorm())
  for nn in range(1, mg4vData.timeSteps + 1):
    mg4v.MultigridCycle()
    l2, linf = mg4v.ResidNorm()
    mg4vData.LogResidualAndError(nn, l2, linf, mg4v.ErrorNorm())
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
  mg4wData.PrintHeaders()
  l2, linf = mg4w.ResidNorm()
  mg4wData.LogResidualAndError(0, l2, linf, mg4w.ErrorNorm())
  for nn in range(1, mg4wData.timeSteps + 1):
    mg4w.MultigridCycle()
    l2, linf = mg4w.ResidNorm()
    mg4wData.LogResidualAndError(nn, l2, linf, mg4w.ErrorNorm())
  print("\n\n")

  # ------------------------------------------------------------------
  # 4 level full multigrid, W cycle
  print("-----------------------------------------")
  print("4 Level Full Multigrid W Cycle")
  print("-----------------------------------------")
  fmg4wData = sd.simulationData(options, 4, "W", "4 Level FMG W")
  # construct grids
  fmg4w = mg.mgSolution(fmg4wData)
  # march solution in time
  fmg4wData.PrintHeaders()  
  l2, linf = fmg4w.ResidNorm()
  fmg4wData.LogResidualAndError(0, l2, linf, fmg4w.ErrorNorm())
  for nn in range(1, fmg4wData.timeSteps + 1):
    if nn == 1:
      fmg4w.FullMultigridCycle()
    else:
      fmg4w.MultigridCycle()
    l2, linf = fmg4w.ResidNorm()
    fmg4wData.LogResidualAndError(nn, l2, linf, fmg4w.ErrorNorm())
  print("\n\n")

  # ------------------------------------------------------------------
  # plot solutions
  _, ax = plt.subplots(2, 3, figsize=(24, 12))
  baseline.PlotExact(ax[0, 0])
  baseline.PlotNode(ax[0, 1], baselineData.name)
  mg2v.PlotNode(ax[0, 2], mg2vData.name)
  mg4v.PlotNode(ax[1, 0], mg4vData.name)
  mg4w.PlotNode(ax[1, 1], mg4wData.name)
  fmg4w.PlotNode(ax[1, 2], fmg4wData.name)
  plt.tight_layout()
  plt.show()

  _, ax = plt.subplots(1, 2, figsize=(24, 12))
  # plot residual vs iteration
  ax[0].set_xlabel("Iteration")
  ax[0].set_ylabel("Normalized Residual")
  ax[0].semilogy(baselineData.iteration,
                    baselineData.NormResids(), "k", lw=3)
  ax[0].semilogy(mg2vData.iteration, mg2vData.NormResids(), "b", lw=3)
  ax[0].semilogy(mg4vData.iteration, mg4vData.NormResids(), "r", lw=3)
  ax[0].semilogy(mg4wData.iteration, mg4wData.NormResids(), "g", lw=3)
  ax[0].semilogy(fmg4wData.iteration, fmg4wData.NormResids(), "c", lw=3)
  ax[0].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4wData.name])
  ax[0].grid(True)
  # plot residual vs wall clock time
  ax[1].set_xlabel("Wall Clock Time (s)")
  ax[1].set_ylabel("Normalized Residual")
  ax[1].semilogy(baselineData.times, baselineData.NormResids(), "k", lw=3)
  ax[1].semilogy(mg2vData.times, mg2vData.NormResids(), "b", lw=3)
  ax[1].semilogy(mg4vData.times, mg4vData.NormResids(), "r", lw=3)
  ax[1].semilogy(mg4wData.times, mg4wData.NormResids(), "g", lw=3)
  ax[1].semilogy(fmg4wData.times, fmg4wData.NormResids(), "c", lw=3)
  ax[1].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4wData.name])
  ax[1].grid(True)
  plt.tight_layout()
  plt.show()

  _, ax = plt.subplots(1, 2, figsize=(24, 12))
  # plot error vs iteration
  ax[0].set_xlabel("Iteration")
  ax[0].set_ylabel("Error Norm")
  ax[0].semilogy(baselineData.iteration,
                    baselineData.error, "k", lw=3)
  ax[0].semilogy(mg2vData.iteration, mg2vData.error, "b", lw=3)
  ax[0].semilogy(mg4vData.iteration, mg4vData.error, "r", lw=3)
  ax[0].semilogy(mg4wData.iteration, mg4wData.error, "g", lw=3)
  ax[0].semilogy(fmg4wData.iteration, fmg4wData.error, "c", lw=3)
  ax[0].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4wData.name])
  ax[0].grid(True)
  # plot error vs wall clock time
  ax[1].set_xlabel("Wall Clock Time (s)")
  ax[1].set_ylabel("Error Norm")
  ax[1].semilogy(baselineData.times, baselineData.error, "k", lw=3)
  ax[1].semilogy(mg2vData.times, mg2vData.error, "b", lw=3)
  ax[1].semilogy(mg4vData.times, mg4vData.error, "r", lw=3)
  ax[1].semilogy(mg4wData.times, mg4wData.error, "g", lw=3)
  ax[1].semilogy(fmg4wData.times, fmg4wData.error, "c", lw=3)
  ax[1].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4wData.name])
  ax[1].grid(True)
  plt.tight_layout()
  plt.show()


  print("-----------------------------------------")
  print("Summary")
  print("-----------------------------------------")
  baselineData.PrintTimeToThreshold()
  mg2vData.PrintTimeToThreshold()
  mg4vData.PrintTimeToThreshold()
  mg4wData.PrintTimeToThreshold()
  fmg4wData.PrintTimeToThreshold()


if __name__ == "__main__":
  main()
