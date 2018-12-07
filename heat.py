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
  parser.add_option("-c", "--cornerTemperature", action="store",
                    dest="cornerTemp",
                    default="500",
                    help="corner temperature to use as BC. " \
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
  print("Iteration        L2 Residual     Linf Residual       Time")
  l2, linf = baseline.ResidNorm()
  baselineData.LogResidual(0, l2, linf)
  for nn in range(1, baselineData.timeSteps + 1):
    baseline.MultigridCycle()
    l2, linf = baseline.ResidNorm()
    baselineData.LogResidual(nn, l2, linf)
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
  print("Iteration        L2 Residual     Linf Residual       Time")
  l2, linf = mg2v.ResidNorm()
  mg2vData.LogResidual(0, l2, linf)
  for nn in range(1, mg2vData.timeSteps + 1):
    mg2v.MultigridCycle()
    l2, linf = mg2v.ResidNorm()
    mg2vData.LogResidual(nn, l2, linf)
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
  print("Iteration        L2 Residual     Linf Residual       Time")
  l2, linf = mg4v.ResidNorm()
  mg4vData.LogResidual(0, l2, linf)
  for nn in range(1, mg4vData.timeSteps + 1):
    mg4v.MultigridCycle()
    l2, linf = mg4v.ResidNorm()
    mg4vData.LogResidual(nn, l2, linf)
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
  print("Iteration        L2 Residual     Linf Residual       Time")
  l2, linf = mg4w.ResidNorm()
  mg4wData.LogResidual(0, l2, linf)
  for nn in range(1, mg4wData.timeSteps + 1):
    mg4w.MultigridCycle()
    l2, linf = mg4w.ResidNorm()
    mg4wData.LogResidual(nn, l2, linf)
  print("\n\n")

  # ------------------------------------------------------------------
  # 4 level full multigrid, V cycle
  print("-----------------------------------------")
  print("4 Level Full Multigrid V Cycle")
  print("-----------------------------------------")
  fmg4vData = sd.simulationData(options, 4, "V", "4 Level FMG V")
  # construct grids
  fmg4v = mg.mgSolution(fmg4vData)
  # march solution in time
  print("Iteration        L2 Residual     Linf Residual       Time")
  l2, linf = fmg4v.ResidNorm()
  fmg4vData.LogResidual(0, l2, linf)
  for nn in range(1, fmg4vData.timeSteps + 1):
    fmg4v.FullMultigridCycle()
    l2, linf = fmg4v.ResidNorm()
    fmg4vData.LogResidual(nn, l2, linf)
  print("\n\n")

  # DEBUG
  #fig1, ax1 = plt.subplots(figsize=(12, 8))
  #plt.xlabel("X (m)")
  #plt.ylabel("Y (m)")
  #plt.title("Residual Contour")
  #cf = ax1.contourf(fmg4v.levels[0].centers[1:-1, 1:-1, 0], \
  #    fmg4v.levels[0].centers[1:-1,1:-1, 1], \
  #    fmg4v.levels[0].CalcResidual())
  #cbar = fig1.colorbar(cf)
  #cbar.ax.set_ylabel("Residual")
  #ax1.grid(True)
  #plt.tight_layout()
  #plt.show()


  # ------------------------------------------------------------------
  # plot solutions
  _, ax = plt.subplots(2, 3, figsize=(24, 12))
  baseline.PlotNode(ax[0, 0], baselineData.name)
  mg2v.PlotNode(ax[0, 1], mg2vData.name)
  mg4v.PlotNode(ax[0, 2], mg4vData.name)
  mg4w.PlotNode(ax[1, 0], mg4wData.name)
  fmg4v.PlotNode(ax[1, 1], fmg4vData.name)
  plt.tight_layout()
  plt.show()

  _, ax = plt.subplots(1, 2, figsize=(24, 12))
  # plot residual vs iteration
  ax[0].set_xlabel("Iteration")
  ax[0].set_ylabel("Normalized Residual")
  ax[0].semilogy(baselineData.iteration, baselineData.NormResids(), "k", lw=3)
  ax[0].semilogy(mg2vData.iteration, mg2vData.NormResids(), "b", lw=3)
  ax[0].semilogy(mg4vData.iteration, mg4vData.NormResids(), "r", lw=3)
  ax[0].semilogy(mg4wData.iteration, mg4wData.NormResids(), "g", lw=3)
  ax[0].semilogy(fmg4vData.iteration, fmg4vData.NormResids(), "c", lw=3)
  ax[0].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4vData.name])
  ax[0].grid(True)
  # plot residual vs wall clock time
  ax[1].set_xlabel("Wall Clock Time (s)")
  ax[1].set_ylabel("Normalized Residual")
  ax[1].semilogy(baselineData.times, baselineData.NormResids(), "k", lw=3)
  ax[1].semilogy(mg2vData.times, mg2vData.NormResids(), "b", lw=3)
  ax[1].semilogy(mg4vData.times, mg4vData.NormResids(), "r", lw=3)
  ax[1].semilogy(mg4wData.times, mg4wData.NormResids(), "g", lw=3)
  ax[1].semilogy(fmg4vData.times, fmg4vData.NormResids(), "c", lw=3)
  ax[1].legend([baselineData.name, mg2vData.name, mg4vData.name,
                   mg4wData.name, fmg4vData.name])
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
  fmg4vData.PrintTimeToThreshold()


if __name__ == "__main__":
  main()
