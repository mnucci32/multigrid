#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 20

def LinearInterp(x0, x1, d0, d1, x):
  diff = x1 - x0
  dist = np.sqrt(np.dot(diff, diff))
  norm = diff / dist
  coeff = np.dot(x - x0, norm)
  di = d0 * (1.0 - coeff) + coeff * d1
  xi = x0 * (1.0 - coeff) + coeff * x1
  return xi, di

def TrilinearInterp(coords, data, x):
  # 4 linear interpolations to convert to 2D
  x0, d0 = LinearInterp(coords[0,:], coords[4,:], data[0,:], data[4,:], x)
  x1, d1 = LinearInterp(coords[1,:], coords[5,:], data[1,:], data[5,:], x)
  x2, d2 = LinearInterp(coords[2,:], coords[6,:], data[2,:], data[6,:], x)
  x3, d3 = LinearInterp(coords[3,:], coords[7,:], data[3,:], data[7,:], x)
  # 2 linear interpolations to convert to 1D
  x4, d4 = LinearInterp(x0, x1, d0, d1, x)
  x5, d5 = LinearInterp(x2, x3, d2, d3, x)
  # 1 linear interpolation to complete trilinear interpolation
  xf, df = LinearInterp(x4, x5, d4, d5, x)
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
    for ss in range(0, self.sweeps):
      # loop over interior solution
      for xx in range(1, self.solution.shape[0] - 1):
        for yy in range(1, self.solution.shape[1] - 1):
          self.solution[xx, yy] = 0.25 * \
              (self.solution[xx - 1, yy] + self.solution[xx + 1, yy] + \
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

  def ResidNorm(self, nn):
    r = self.levels[0].residual
    resid = np.sum(r * r) / (r.shape[0] * r.shape[1])
    print(nn, resid)

