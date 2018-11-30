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
  coeff = np.dot(x - x0, norm) / dist
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

def GaussSeidel(matrix, rhs, sweeps):
  # matrix BCs should already be set, so loop over interior only
  for _ in range(0, sweeps):
    # loop over interior solution
    for xx in range(1, matrix.shape[0] - 1):
      for yy in range(1, matrix.shape[1] - 1):
        matrix[xx, yy] = 0.25 * (rhs[xx, yy] + \
            matrix[xx - 1, yy] + matrix[xx + 1, yy] + \
            matrix[xx, yy - 1] + matrix[xx, yy + 1])
  return matrix

def Laplacian(solution, xx, yy):
  return 4.0 * solution[xx,yy] - solution[xx-1,yy] - \
      solution[xx + 1, yy] - solution[xx, yy - 1] - \
      solution[xx, yy + 1]

def Residual(sol, forcing, nu, area):
  residual = np.zeros(forcing.shape)
  # loop over cells and calculate residual
  for xx in range(0, residual.shape[0]):
    for yy in range(0, residual.shape[1]):
      residual[xx, yy] = forcing[xx, yy] - \
          nu * Laplacian(sol, xx + 1, yy + 1) / area
  return residual

def CellsToNodes(cells, haveGhosts):
  # initialize nodal data
  if haveGhosts:
    nx = cells.shape[0] - 1
    ny = cells.shape[1] - 1
  else:
    nx = cells.shape[0] + 1
    ny = cells.shape[1] + 1
  nodes = np.zeros((nx, ny))
  if not haveGhosts:
    # assign corners
    nodes[0, 0] = cells[0, 0]
    nodes[-1, 0] = cells[-1, 0]
    nodes[0, -1] = cells[0, -1]
    nodes[-1, -1] = cells[-1, -1]
    # assign edges
    for xx in range(1, nx - 1):
      nodes[xx, 0] = 0.5 * (cells[xx, 0] + cells[xx - 1, 0])
      nodes[xx, -1] = 0.5 * (cells[xx, -1] + cells[xx - 1, -1])
    for yy in range(1, ny - 1):
      nodes[0, yy] = 0.5 * (cells[0, yy] + cells[0, yy - 1])
      nodes[-1, yy] = 0.5 * (cells[-1, yy] + cells[-1, yy - 1])
    # loop over interior nodes
    for xx in range(1, nx - 1):
      for yy in range(1, ny - 1):
        nodes[xx, yy] = 0.25 * (cells[xx - 1, yy] + cells[xx, yy] + \
            cells[xx - 1, yy - 1] + cells[xx, yy - 1])
  else:
    # loop over all nodes
    for xx in range(1, nx + 1):
      for yy in range(1, ny + 1):
        nodes[xx - 1, yy - 1] = 0.25 * (cells[xx - 1, yy] + cells[xx, yy] + \
            cells[xx - 1, yy - 1] + cells[xx, yy - 1])
  return nodes


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
    self.solution = np.zeros((xNum + 1, yNum + 1))
    self.forcing = np.zeros((xNum - 1, yNum - 1))

  def NumNodes(self):
    return self.numNodesX * self.numNodesY

  def NumCells(self):
    return (self.numNodesX - 1) * (self.numNodesY - 1)

  def AssignBCs(self, data, isSolution):
    if isSolution:
      return self.AssignSolutionBCs(data)
    else:
      return self.AssignCorrectionBCs(data)

  def AssignCorrectionBCs(self, corr):
    rhs = self.Rhs()
    corr[0,:] = rhs[0,:]
    corr[-1,:] = rhs[-1,:]
    corr[:, 0] = rhs[:, 0]
    corr[:,-1] = rhs[:,-1]
    return corr

  def AssignSolutionBCs(self, sol):
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

    sol[0, 1:-1] = xl
    sol[-1, 1:-1] = xu
    sol[1:-1, 0] = yl
    sol[1:-1, -1] = yu
    return sol

  def Rhs(self):
    rhs = np.zeros((self.forcing.shape[0] + 2, self.forcing.shape[1] + 2))
    rhs[1:-1, 1:-1] = self.area * self.forcing
    # assign edges
    rhs[0, 1:-1] = rhs[1, 1:-1]
    rhs[-1, 1:-1] = rhs[-2, 1:-1]
    rhs[1:-1, 0] = rhs[1:-1, 1]
    rhs[1:-1, -1] = rhs[1:-1, -2]
    # assign corners
    rhs[0, 0] = 0.5 * (rhs[0, 1] + rhs[1, 0])
    rhs[-1, -1] = 0.5 * (rhs[-1, -2] + rhs[-2, -1])
    rhs[0, -1] = 0.5 * (rhs[0, -2] + rhs[1, -1])
    rhs[-1, 0] = 0.5 * (rhs[-2, 0] + rhs[-1, 1])
    return rhs

  def CalcResidual(self):
    # assign BCs
    self.solution = self.AssignBCs(self.solution, True)
    return Residual(self.solution, self.forcing, self.nu, self.area)

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
    cf = ax.contourf(self.coords[:,:, 0], self.coords[:,:, 1], nodalSolution, \
        levels=np.linspace(np.min(nodalSolution), np.max(nodalSolution), 11))
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
    self.sweeps = 5
    self.preRelaxationSweeps = 2
    self.postRelaxationSweeps = 1
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
    r = self.levels[0].CalcResidual()[:]
    resid = np.linalg.norm(r) / np.sqrt(len(r))
    dt = time.time() - t0
    print("{0:5d} {1:22.4e} {2:15.4e}".format(nn, resid, dt))

  def Restriction(self, ll, residual):
    # fine to coarse transfer
    # full weighting of cells
    # integral preserving so area factor is needed
    factor = self.levels[ll].area / self.levels[ll + 1].area
    for xx in range(0, self.levels[ll + 1].forcing.shape[0]):
      for yy in range(0, self.levels[ll + 1].forcing.shape[1]):
        self.levels[ll + 1].forcing[xx, yy] = factor * 0.25 * \
            (residual[2 * xx, 2 * yy] + \
            residual[2 * xx + 1, 2 * yy] + \
            residual[2 * xx, 2 * yy + 1] + \
            residual[2 * xx + 1, 2 * yy + 1])

  def Prolongation(self, ll, coarseCorrection, fine):
    # coarse to fine transfer
    # solves the error equation - coarse grid error "prolonged" to fine grid
    # convert solution to node
    cc = CellsToNodes(coarseCorrection, True)
    correction = np.zeros(self.levels[ll - 1].solution.shape)
    # use bilinear interpolation
    for xx in range(1, correction.shape[0] - 1):
      for yy in range(1, correction.shape[1] - 1):
        xc = (xx - 1) // 2
        yc = (yy - 1) // 2
        correction[xx, yy] = BilinearInterp(\
            self.levels[ll].coords[xc, yc], \
            self.levels[ll].coords[xc + 1, yc], \
            self.levels[ll].coords[xc, yc + 1], \
            self.levels[ll].coords[xc + 1, yc + 1], \
            cc[xc, yc], cc[xc + 1, yc], cc[xc, yc + 1], cc[xc + 1, yc + 1], \
            self.levels[ll - 1].centers[xx - 1, yy - 1])
    fine += correction
    return fine
    
  def CycleAtLevel(self, fl, sol, isSolution):
    self.levels[fl].AssignBCs(sol, isSolution)
    rhs = self.levels[fl].Rhs()

    if fl == self.numLevel - 1:
      # at coarsest level - recursive base case
      sol = GaussSeidel(sol, rhs, self.sweeps)
      #if isSolution:
      #  self.levels[fl].solution = sol.copy()
    else:
      # pre-relaxation at fine level
      sol = GaussSeidel(sol, rhs, self.preRelaxationSweeps)
      #if isSolution:
      #  self.levels[fl].solution = sol.copy()

      # coarse grid correction
      r = Residual(sol, self.levels[fl].forcing, \
          self.levels[fl].nu, self.levels[fl].area)
      self.Restriction(fl, r)

      # recursive call to next coarse level
      cl = fl + 1
      coarseCorrection = np.zeros((self.levels[cl].solution.shape))
      coarseCorrection = self.CycleAtLevel(cl, coarseCorrection, False)

      # interpolate coarse level correction
      sol = self.Prolongation(cl, coarseCorrection, sol)
      
      # post-relaxation at fine level
      sol = GaussSeidel(sol, rhs, self.postRelaxationSweeps)
      #if isSolution:
      #  self.levels[fl].solution = sol.copy()

    return sol


  def MultigridCycle(self):
    self.CycleAtLevel(0, self.levels[0].solution, True)


