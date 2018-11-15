#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams["font.size"] = 20

def LinearInterp(x0, x1, d0, d1, x):
  diff = x1 - x0
  dist = np.sqrt(np.dot(diff, diff))
  norm = diff / dist
  coeff = np.dot(x - x0, norm)
  di = d0 * (1.0 - coeff) + coeff * d1
  xi = x0 * (1.0 - coeff) + coeff * x1
  return xi, di

def BilinearInterp(x0, x1, x2, x3, d0, d1, d2, d3, x):
  # 2 linear interpolations to convert to 1D
  x4, d4 = LinearInterp(x0, x1, d0, d1, x)
  x5, d5 = LinearInterp(x2, x3, d2, d3, x)
  # 1 linear interpolation to complete trilinear interpolation
  _, df = LinearInterp(x4, x5, d4, d5, x)
  return df


class gridLevel:
  def __init__(self, xRange, xNum, yRange, yNum, nu):
    xc = np.linspace(xRange[0], xRange[1], xNum)
    yc = np.linspace(yRange[0], yRange[1], yNum)
    self.numNodesX = xNum
    self.numNodesY = yNum
    self.coords = np.zeros((xNum, yNum, 2))
    for xx in range(0, xNum):
      for yy in range(0, yNum):
        self.coords[xx, yy, 0] = xc[xx]
        self.coords[xx, yy, 1] = yc[yy]
    self.dx = xc[1] - xc[0]
    self.dy = yc[1] - yc[0]
    self.area = self.dx * self.dy
    self.centers = np.zeros((xNum - 1, yNum - 1, 2))
    for xx in range(0, xNum - 1):
      for yy in range(0, yNum - 1):
        self.centers[xx, yy, 0] = xc[xx] + self.dx / 2.0
        self.centers[xx, yy, 1] = yc[yy] + self.dy / 2.0
    self.nu = nu
    self.dt = self.area / (4.0 * self.nu)
    self.xLower = np.linspace(100.0, 150.0, yNum)
    self.xUpper = np.linspace(150.0, 200.0, yNum)
    self.yLower = np.linspace(100.0, 150.0, xNum)
    self.yUpper = np.linspace(150.0, 200.0, xNum)
    self.sweeps = 5
    self.solution = np.zeros((xNum + 1, yNum + 1))
    self.residual = np.zeros((xNum - 1, yNum - 1))
    self.forcing = np.zeros((xNum - 1, yNum - 1))

  def NumNodes(self):
    return self.numNodesX * self.numNodesY

  def NumCells(self):
    return (self.numNodesX - 1) * (self.numNodesY - 1)

  def Laplacian(self, xx, yy):
    return -4.0 * self.solution[xx,yy] + self.solution[xx-1,yy] + \
        self.solution[xx + 1, yy] + self.solution[xx, yy - 1] + \
        self.solution[xx, yy + 1]
            
  def AssignBCs(self):
    xl = np.zeros((self.numNodesY - 1))
    xu = np.zeros((self.numNodesY - 1))
    for ii in range(0, len(xl)):
      xl[ii] = 0.5 * (self.xLower[ii] + self.xLower[ii + 1])
      xu[ii] = 0.5 * (self.xUpper[ii] + self.xUpper[ii + 1])
    yl = np.zeros((self.numNodesX - 1))
    yu = np.zeros((self.numNodesX - 1))
    for ii in range(0, len(yl)):
      yl[ii] = 0.5 * (self.yLower[ii] + self.yLower[ii + 1])
      yu[ii] = 0.5 * (self.yUpper[ii] + self.yUpper[ii + 1])

    self.solution[0, 1:-1] = xl
    self.solution[-1, 1:-1] = xu
    self.solution[1:-1, 0] = yl
    self.solution[1:-1, -1] = yu

  def GaussSeidel(self):
    # assign boundary conditions
    self.AssignBCs()
    for _ in range(0, self.sweeps):
      # loop over interior solution
      for xx in range(1, self.solution.shape[0] - 1):
        for yy in range(1, self.solution.shape[1] - 1):
          self.solution[xx, yy] = 0.25 * (self.area * \
              self.forcing[xx - 1, yy - 1] + 
              self.solution[xx - 1, yy] + self.solution[xx + 1, yy] + \
              self.solution[xx, yy - 1] + self.solution[xx, yy + 1])


  def CalcResidual(self):
    # assign boundary conditions
    self.AssignBCs()
    # loop over cells and calculate residual
    self.residual = np.zeros(self.residual.shape)
    for xx in range(0, self.residual.shape[0]):
      for yy in range(0, self.residual.shape[1]):
        self.residual[xx, yy] = self.nu * self.Laplacian(xx + 1, yy + 1) \
            / self.area

  def ToNodes(self):
    # initialize nodal solution
    nodalSolution = np.zeros((self.numNodesX, self.numNodesY))
    # assign boundary conditions
    nodalSolution[0,:] = self.xLower
    nodalSolution[-1,:] = self.xUpper
    nodalSolution[:, 0] = self.yLower
    nodalSolution[:, -1] = self.yUpper
    # loop over interior nodes
    for xx in range(1, self.numNodesX - 1):
      for yy in range(1, self.numNodesY - 1):
        nodalSolution[xx, yy] = 0.25 * (self.solution[xx + 1, yy] + \
            self.solution[xx, yy] + self.solution[xx + 1, yy + 1] + \
            self.solution[xx, yy + 1])
    return nodalSolution

  def ResidualToNodes(self):
    # initialize nodal residual
    nodalResidual = np.zeros((self.numNodesX, self.numNodesY))
    # assign corners
    nodalResidual[0, 0] = self.residual[0, 0]
    nodalResidual[-1, 0] = self.residual[-1, 0]
    nodalResidual[0, -1] = self.residual[0, -1]
    nodalResidual[-1, -1] = self.residual[-1, -1]
    # assign edges
    for xx in range(1, self.numNodesX - 1):
      nodalResidual[xx, 0] = 0.5 * (self.residual[xx, 0] + \
          self.residual[xx - 1, 0])
      nodalResidual[xx, -1] = 0.5 * (self.residual[xx, -1] + \
          self.residual[xx - 1, -1])
    for yy in range(1, self.numNodesY - 1):
      nodalResidual[0, yy] = 0.5 * (self.residual[0, yy] + \
          self.residual[0, yy - 1])
      nodalResidual[-1, yy] = 0.5 * (self.residual[-1, yy] + \
          self.residual[-1, yy - 1])
    # loop over interior nodes
    for xx in range(1, self.numNodesX - 1):
      for yy in range(1, self.numNodesY - 1):
        nodalResidual[xx, yy] = 0.25 * (self.residual[xx + 1, yy] + \
            self.residual[xx, yy] + self.residual[xx + 1, yy + 1] + \
            self.residual[xx, yy + 1])
    return nodalResidual

  def PlotCenter(self):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Temperature Contour")
    cf = ax.contourf(self.centers[:,:, 0], self.centers[:,:, 1], \
        self.solution[1:-1,1:-1])
    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel("Temperature (K)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

  def PlotNode(self):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Temperature Contour")
    nodalSolution = self.ToNodes()
    cf = ax.contourf(self.coords[:,:, 0], self.coords[:,:, 1], nodalSolution)
    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel("Temperature (K)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

  def Print(self):
    print("CELL CENTER SOLUTION")
    print(self.solution)
    print("NODAL SOLUTION")
    print(self.ToNodes())

class mgSolution:
  def __init__(self, xRange, xNum, yRange, yNum, nu, levels):
    self.numLevel = levels
    self.levels = []
    for ll in range(0, levels):
      xn = xNum
      yn = yNum
      if ll > 0:
        xn = xNum // 2 ** ll + 1
        yn = yNum // 2 ** ll + 1
      grid = gridLevel(xRange, xn, yRange, yn, nu)
      self.levels.append(grid)

  def PlotCenter(self):
    self.levels[0].PlotCenter()

  def PlotNode(self):
    self.levels[0].PlotNode()

  def Print(self):
    self.levels[0].Print()

  def ResidNorm(self, nn, t0):
    r = self.levels[0].residual
    resid = np.sqrt(np.sum(r * r) / len(r[:]))
    dt = time.time() - t0
    print("{0:5d} {1:22.4e} {2:15.4e}".format(nn, resid, dt))

  def Restriction(self, ll):
    # fine to coarse transfer
    # full weighting of cells
    for xx in range(0, self.levels[ll + 1].forcing.shape[0]):
      for yy in range(0, self.levels[ll + 1].forcing.shape[1]):
        self.levels[ll + 1].forcing[xx, yy] = 0.25 * \
            (self.levels[ll].residual[2 * xx, 2 * yy] + \
            self.levels[ll].residual[2 * xx + 1, 2 * yy] + \
            self.levels[ll].residual[2 * xx, 2 * yy + 1] + \
            self.levels[ll].residual[2 * xx + 1, 2 * yy + 1])

  def Prolongation(self, ll):
    # coarse to fine transfer
    # convert residual to node
    rc = self.levels[ll].ResidualToNodes()
    # use bilinear interp
    for xx in range(0, self.levels[ll - 1].forcing.shape[0]):
      for yy in range(0, self.levels[ll - 1].forcing.shape[1]):
        xc = xx // 2
        yc = yy // 2
        self.levels[ll - 1].forcing = BilinearInterp(\
            self.levels[ll].coords[xc, yc], self.levels[ll].coords[xc + 1, yc], \
            self.levels[ll].coords[xc, yc + 1], \
            self.levels[ll].coords[xc + 1, yc + 1],
            rc[xc, yc], rc[xc + 1, yc], rc[xc, yc + 1], rc[xc + 1, yc + 1], \
            self.levels[ll - 1].centers[xx, yy])
    

  def MultigridCycle(self):
    self.levels[0].GaussSeidel()
    self.levels[0].CalcResidual()

