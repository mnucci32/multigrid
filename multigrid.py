#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams["font.size"] = 20

def Distance(a, b):
  return np.linalg.norm(a - b)

def LinearInterpCoeff(x0, x1, x):
  diff = x1 - x0
  rdist = 1.0 / np.sqrt(np.dot(diff, diff))
  norm = diff * rdist
  return np.dot(x - x0, norm) * rdist

def BilinearInterpCoeffs(x0, x1, x2, x3, x):
  coeffs = np.zeros((3))
  coeffs[0] = LinearInterpCoeff(x0, x1, x)
  x4 = LinearInterp(x0, x1, coeffs[0])
  coeffs[1] = LinearInterpCoeff(x2, x3, x)
  x5 = LinearInterp(x2, x3, coeffs[1])
  coeffs[2] = LinearInterpCoeff(x4, x5, x)
  return coeffs

def LinearInterp(x0, x1, coeff):
  return x0 * (1.0 - coeff) + coeff * x1

def BilinearInterp(d0, d1, d2, d3, coeffs):
  # 2 linear interpolations to convert to 1D
  d4 = LinearInterp(d0, d1, coeffs[0])
  d5 = LinearInterp(d2, d3, coeffs[1])
  # 1 linear interpolation to complete trilinear interpolation
  df = LinearInterp(d4, d5, coeffs[2])
  return df

def QuadraticInterp(x0, x1, x2, d0, d1, d2, x, i):
  c0 = ((x[i] - x1[i]) * (x[i] - x2[i])) / ((x0[i] - x1[i]) * (x0[i] - x2[i]))
  c1 = ((x[i] - x0[i]) * (x[i] - x2[i])) / ((x1[i] - x0[i]) * (x1[i] - x2[i]))
  c2 = ((x[i] - x0[i]) * (x[i] - x1[i])) / ((x2[i] - x0[i]) * (x2[i] - x1[i]))
  if i == 0:
    loc = np.array([x[i], x0[1]])
  else:
    loc = np.array([x0[0], x[i]])
  return c0 * d0 + c1 * d1 + c2 * d2, loc

def BiquadraticInterp(xsw, xs, xse, xw, xc, xe, xnw, xn, xne, sw, s, se, w, c, \
    e, nw, n, ne, x):
  lower, xl = QuadraticInterp(xsw, xs, xse, sw, s, se, x, 0)
  middle, xm = QuadraticInterp(xw, xc, xe, w, c, e, x, 0)
  upper, xu = QuadraticInterp(xnw, xn, xne, nw, n, ne, x, 0)
  interp, _ = QuadraticInterp(xl, xm, xu, lower, middle, upper, x, 1)
  return interp

def InverseDistanceInterp(coords, data, x):
  p = 12
  denominator = 0.0
  numerator = 0.0
  for ii in range(0, coords.shape[0]):
    diff = coords[ii,:] - x
    denomPart = np.power(np.dot(diff, diff), p / 2)
    denominator += 1.0 / denomPart
    numerator += data[ii] / denomPart
  return numerator / denominator

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

def HeatFunction(relX, relY, temp):
  # exact solution from http://www.mgnet.org/mgnet/tutorials/xwb/example.html
  a = np.sinh(np.pi) * np.sin(np.pi / 2)
  return temp / a * np.sinh(np.pi * relX) * np.sin(np.pi * relY)


class gridLevel:
  def __init__(self, xc, yc, nu, maxTemp):
    xNum = len(xc)
    yNum = len(yc)
    self.numNodesX = xNum
    self.numNodesY = yNum
    self.anchor = np.array([xc[0], yc[0]])
    self.relCoords = np.zeros((xNum, yNum, 2))
    self.coords = np.zeros((xNum, yNum, 2))
    for xx in range(0, xNum):
      for yy in range(0, yNum):
        self.coords[xx, yy, :] = [xc[xx], yc[yy]]
        self.relCoords[xx, yy,:] = self.coords[xx, yy,:] - self.anchor
    xlen = xc[-1] - xc[0]
    ylen = yc[-1] - yc[0]
    self.relCoords[:, :, 0] /= xlen
    self.relCoords[:, :, 1] /= ylen
    self.dx = xc[1] - xc[0]
    self.dy = yc[1] - yc[0]
    self.area = self.dx * self.dy
    self.centers = np.zeros((xNum + 1, yNum + 1, 2))
    self.relCenters = np.zeros((xNum + 1, yNum + 1, 2))
    xcc = np.linspace(xc[0] - self.dx / 2.0, xc[-1] + self.dx / 2.0, xNum + 1)
    ycc = np.linspace(yc[0] - self.dy / 2.0, yc[-1] + self.dy / 2.0, yNum + 1)
    for xx in range(0, xNum + 1):
      for yy in range(0, yNum + 1):
        self.centers[xx, yy, :] = [xcc[xx], ycc[yy]]
        self.relCenters[xx, yy,:] = self.centers[xx, yy,:] - self.anchor
    self.relCenters[:,:, 0] /= xlen
    self.relCenters[:,:, 1] /= ylen
    self.nu = nu
    self.maxTemp = maxTemp
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

  def Relax(self, sweeps):
    self.solution = GaussSeidel(self.solution, self.Rhs(), sweeps)

  def ToNodes(self):
    # initialize nodal solution
    nodalSolution = np.zeros((self.numNodesX, self.numNodesY))
    # assign boundary conditions
    nodalSolution[0, :] = HeatFunction(
        self.relCoords[0, :, 0], self.relCoords[0, :, 1], self.maxTemp)
    nodalSolution[-1, :] = HeatFunction(self.relCoords[-1, :, 0],
                                        self.relCoords[-1, :, 1], self.maxTemp)
    nodalSolution[:, 0] = HeatFunction(
        self.relCoords[:, 0, 0], self.relCoords[:, 0, 1], self.maxTemp)
    nodalSolution[:, -1] = HeatFunction(self.relCoords[:, -1, 0],
                                        self.relCoords[:, -1, 1], self.maxTemp)
    # loop over interior nodes
    for xx in range(1, self.numNodesX - 1):
      for yy in range(1, self.numNodesY - 1):
        nodalSolution[xx, yy] = 0.25 * (self.solution[xx + 1, yy] + \
            self.solution[xx, yy] + self.solution[xx + 1, yy + 1] + \
            self.solution[xx, yy + 1])
    return nodalSolution

  def EdgeSolution(self, xx, yy):
    # assign boundary conditions
    north = 0.5 * (self.solution[xx, yy] + self.solution[xx, yy + 1])
    south = 0.5 * (self.solution[xx, yy] + self.solution[xx, yy - 1])
    east = 0.5 * (self.solution[xx, yy] + self.solution[xx + 1, yy])
    west = 0.5 * (self.solution[xx, yy] + self.solution[xx - 1, yy])
    return north, south, east, west

  def EdgeCoords(self, xx, yy):
    # assign boundary conditions
    north = 0.5 * (self.centers[xx, yy, :] + self.centers[xx, yy + 1, :])
    south = 0.5 * (self.centers[xx, yy, :] + self.centers[xx, yy - 1, :])
    east = 0.5 * (self.centers[xx, yy, :] + self.centers[xx + 1, yy, :])
    west = 0.5 * (self.centers[xx, yy, :] + self.centers[xx - 1, yy, :])
    return north, south, east, west

  def PlotCenter(self):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Temperature Contour")
    cf = ax.contourf(self.centers[1:-1,1:-1, 0], self.centers[1:-1,1:-1, 1], \
        self.solution[1:-1,1:-1])
    cbar = fig.colorbar(cf)
    cbar.ax.set_ylabel("Temperature (K)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

  def PlotNode(self, ax, title):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    nodalSolution = self.ToNodes()
    cf = ax.contourf(self.coords[:,:, 0], self.coords[:,:, 1], nodalSolution, \
        levels=np.linspace(np.min(nodalSolution), np.max(nodalSolution), 11))
    cbar = plt.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel("Temperature (K)")
    ax.grid(True)

  def PlotExact(self, ax):
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Exact")
    nodalSolution = HeatFunction(
        self.relCoords[:, :, 0], self.relCoords[:, :, 1], self.maxTemp)
    cf = ax.contourf(self.coords[:,:, 0], self.coords[:,:, 1], nodalSolution, \
        levels=np.linspace(np.min(nodalSolution), np.max(nodalSolution), 11))
    cbar = plt.colorbar(cf, ax=ax)
    cbar.ax.set_ylabel("Temperature (K)")
    ax.grid(True)

  def Print(self):
    print("CELL CENTER SOLUTION")
    print(self.solution)
    print("NODAL SOLUTION")
    print(self.ToNodes())

  def Ax(self):
    prod = np.zeros(self.forcing.shape)
    for xx in range(0, prod.shape[0]):
      for yy in range(0, prod.shape[1]):
        prod[xx, yy] = self.nu * \
            Laplacian(self.solution, xx + 1, yy + 1) / self.area
    return prod

class mgSolution:
  def __init__(self, simData):
    self.numLevel = simData.gridLevels
    self.cycleType = simData.cycleType
    self.sweeps = 5
    self.preRelaxationSweeps = 2
    self.postRelaxationSweeps = 1
    xNum = len(simData.xc)
    yNum = len(simData.yc)
    self.levels = []
    for ll in range(0, simData.gridLevels):
      xn = xNum
      yn = yNum
      if ll > 0:
        xn = xNum // 2 ** ll + 1
        yn = yNum // 2 ** ll + 1
      xc = np.linspace(simData.xc[0], simData.xc[-1], xn)
      yc = np.linspace(simData.yc[0], simData.yc[-1], yn)
      grid = gridLevel(xc, yc, simData.nu, simData.maxTemp)
      self.levels.append(grid)
    self.prolongationCoeffs = []
    for ll in range(0, simData.gridLevels - 1):
      coeffs = np.zeros((self.levels[ll].centers.shape[0],
                         self.levels[ll].centers.shape[0], 3))
      # use bilinear interpolation coeffs
      for xx in range(1, coeffs.shape[0] - 1):
        for yy in range(1, coeffs.shape[1] - 1):
          xc = (xx - 1) // 2
          yc = (yy - 1) // 2
          coeffs[xx, yy] = BilinearInterpCoeffs(\
              self.levels[ll + 1].coords[xc, yc], \
              self.levels[ll + 1].coords[xc + 1, yc], \
              self.levels[ll + 1].coords[xc, yc + 1], \
              self.levels[ll + 1].coords[xc + 1, yc + 1], \
              self.levels[ll].centers[xx, yy])
      self.prolongationCoeffs.append(coeffs)

  def CycleIndex(self):
    ind = 1
    if self.cycleType == "W":
      ind = 2
    return ind

  def PlotCenter(self):
    self.levels[0].PlotCenter()

  def PlotNode(self, ax, title):
    self.levels[0].PlotNode(ax, title)

  def PlotExact(self, ax):
    self.levels[0].PlotExact(ax)

  def Print(self):
    self.levels[0].Print()

  def ResidNorm(self):
    r = self.levels[0].CalcResidual()[:]
    resid = np.linalg.norm(r) / np.sqrt(len(r))
    return resid, np.max(np.abs(r))

  def Restriction(self, ll):
    # fine to coarse transfer
    # full weighting of cells
    # integral preserving so area factor is needed
    factor = self.levels[ll].area / self.levels[ll + 1].area
    fine = self.levels[ll].CalcResidual()
    coarse = np.zeros(self.levels[ll + 1].forcing.shape)
    for xx in range(0, coarse.shape[0]):
      for yy in range(0, coarse.shape[1]):
        coarse[xx, yy] = factor * 0.25 * \
            (fine[2 * xx, 2 * yy] + fine[2 * xx + 1, 2 * yy] +
             fine[2 * xx, 2 * yy + 1] + fine[2 * xx + 1, 2 * yy + 1])
    return coarse

  def RestrictionSol(self, ll):
    # fine to coarse transfer
    # full weighting of cells
    coarse = np.zeros(self.levels[ll + 1].solution.shape)
    for xx in range(1, coarse.shape[0] - 1):
      for yy in range(1, coarse.shape[1] - 1):
        coarse[xx, yy] = 0.25 * \
            (self.levels[ll].solution[2 * xx, 2 * yy] + \
            self.levels[ll].solution[2 * xx - 1, 2 * yy] + \
            self.levels[ll].solution[2 * xx, 2 * yy - 1] + \
            self.levels[ll].solution[2 * xx - 1, 2 * yy - 1])
    coarse = self.levels[ll + 1].AssignBCs(coarse, True)
    return coarse

  def Prolongation(self, ll, coarseCorrection, cummulative):
    # coarse to fine transfer
    # solves the error equation - coarse grid error "prolonged" to fine grid
    # convert solution to node
    if cummulative:
      cc = CellsToNodes(coarseCorrection, True)
    else:
      cc = coarseCorrection
    correction = np.zeros(self.levels[ll - 1].solution.shape)
    # use bilinear interpolation
    for xx in range(1, correction.shape[0] - 1):
      for yy in range(1, correction.shape[1] - 1):
        xc = (xx - 1) // 2
        yc = (yy - 1) // 2
        correction[xx, yy] = BilinearInterp(\
            cc[xc, yc], cc[xc + 1, yc], cc[xc, yc + 1], cc[xc + 1, yc + 1], \
            self.prolongationCoeffs[ll - 1][xx, yy, :])
    if cummulative:
      self.levels[ll - 1].solution += correction
    else:
      self.levels[ll - 1].solution = correction
      self.levels[ll - 1].AssignBCs(self.levels[ll - 1].solution, True)

  def HighOrderInterp(self, cl):
    fl = cl - 1
    cNodes = self.levels[cl].ToNodes()
    for xx in range(1, self.levels[fl].solution.shape[0] - 1):
      for yy in range(1, self.levels[fl].solution.shape[1] - 1):
        xc = (xx - 1) // 2
        yc = (yy - 1) // 2
        north, south, east, west = self.levels[cl].EdgeSolution(xc, yc)
        pNorth, pSouth, pEast, pWest = self.levels[cl].EdgeCoords(xc, yc)
        center = self.levels[cl].solution[xc, yc]
        pCenter = self.levels[cl].centers[xc, yc, :]
        sw = cNodes[xc, yc]
        se = cNodes[xc + 1, yc]
        nw = cNodes[xc, yc + 1]
        ne = cNodes[xc + 1, yc + 1]
        pSw = self.levels[cl].coords[xc, yc, :]
        pSe = self.levels[cl].coords[xc + 1, yc, :]
        pNw = self.levels[cl].coords[xc, yc + 1, :]
        pNe = self.levels[cl].coords[xc + 1, yc + 1, :]
        quad = BiquadraticInterp(pSw, pSouth, pSe, \
            pWest, pCenter, pEast, pNw, pNorth, pNe, sw, south, se, west, \
            center, east, nw, north, ne, self.levels[fl].centers[xx, yy,:])
        self.levels[fl].solution[xx, yy] = quad
    self.levels[fl].AssignBCs(self.levels[fl].solution, True)

  def HighOrderInterp2(self, cl):
    fl = cl - 1
    for xx in range(1, self.levels[fl].solution.shape[0] - 1):
      for yy in range(1, self.levels[fl].solution.shape[1] - 1):
        xc = (xx - 1) // 2
        yc = (yy - 1) // 2

        center = self.levels[cl].solution[xc, yc]
        pCenter = self.levels[cl].centers[xc, yc,:]
        west = self.levels[cl].solution[xc - 1, yc]
        pWest = self.levels[cl].centers[xc - 1, yc, :]
        east = self.levels[cl].solution[xc + 1, yc]
        pEast = self.levels[cl].centers[xc + 1, yc, :]

        south = self.levels[cl].solution[xc, yc - 1]
        pSouth = self.levels[cl].centers[xc, yc - 1,:]
        sw = self.levels[cl].solution[xc - 1, yc - 1]
        pSw = self.levels[cl].centers[xc - 1, yc - 1, :]
        se = self.levels[cl].solution[xc + 1, yc - 1]
        pSe = self.levels[cl].centers[xc + 1, yc - 1, :]

        north = self.levels[cl].solution[xc, yc + 1]
        pNorth = self.levels[cl].centers[xc, yc + 1,:]
        nw = self.levels[cl].solution[xc - 1, yc + 1]
        pNw = self.levels[cl].centers[xc - 1, yc + 1, :]
        ne = self.levels[cl].solution[xc + 1, yc + 1]
        pNe = self.levels[cl].centers[xc + 1, yc + 1, :]

        quad = BiquadraticInterp(pSw, pSouth, pSe, \
            pWest, pCenter, pEast, pNw, pNorth, pNe, sw, south, se, west, \
            center, east, nw, north, ne, self.levels[fl].centers[xx, yy,:])
        self.levels[fl].solution[xx, yy] = quad
    self.levels[fl].AssignBCs(self.levels[fl].solution, True)

  def InverseDistanceInterp(self, cl):
    fl = cl - 1
    for xx in range(1, self.levels[fl].solution.shape[0] - 1):
      for yy in range(1, self.levels[fl].solution.shape[1] - 1):
        xc = (xx - 1) // 2
        yc = (yy - 1) // 2

        coords = np.zeros((9, 2))
        coords[0,:] = self.levels[cl].centers[xc - 1, yc - 1,:]
        coords[1,:] = self.levels[cl].centers[xc, yc - 1,:]
        coords[2,:] = self.levels[cl].centers[xc + 1, yc - 1,:]
        coords[3,:] = self.levels[cl].centers[xc - 1, yc,:]
        coords[4,:] = self.levels[cl].centers[xc, yc,:]
        coords[5,:] = self.levels[cl].centers[xc + 1, yc,:]
        coords[6,:] = self.levels[cl].centers[xc - 1, yc + 1,:]
        coords[7,:] = self.levels[cl].centers[xc, yc + 1,:]
        coords[8,:] = self.levels[cl].centers[xc + 1, yc + 1,:]

        data = np.zeros((9))
        data[0] = self.levels[cl].solution[xc - 1, yc - 1]
        data[1] = self.levels[cl].solution[xc, yc - 1]
        data[2] = self.levels[cl].solution[xc + 1, yc - 1]
        data[3] = self.levels[cl].solution[xc - 1, yc]
        data[4] = self.levels[cl].solution[xc, yc]
        data[5] = self.levels[cl].solution[xc + 1, yc]
        data[6] = self.levels[cl].solution[xc - 1, yc + 1]
        data[7] = self.levels[cl].solution[xc, yc + 1]
        data[8] = self.levels[cl].solution[xc + 1, yc + 1]

        invDist = InverseDistanceInterp(coords, data, \
            self.levels[fl].centers[xx, yy,:])
        self.levels[fl].solution[xx, yy] = invDist
    self.levels[fl].AssignBCs(self.levels[fl].solution, True)


  def CycleAtLevel(self, fl):
    self.levels[fl].AssignBCs(self.levels[fl].solution, True)

    if fl == self.numLevel - 1:
      # at coarsest level - recursive base case
      self.levels[fl].Relax(self.sweeps)
    else:
      # pre-relaxation at fine level
      self.levels[fl].Relax(self.preRelaxationSweeps)

      # coarse grid correction
      cl = fl + 1
      self.levels[cl].solution = self.RestrictionSol(fl)
      self.levels[cl].forcing = self.levels[cl].Ax() + self.Restriction(fl)
      solCoarse = self.levels[cl].solution.copy()

      # recursive call to next coarse level
      for _ in range(0, self.CycleIndex()):
        self.CycleAtLevel(cl)
      coarseCorrection = self.levels[cl].solution - solCoarse

      # interpolate coarse level correction
      self.Prolongation(cl, coarseCorrection, True)
      
      # post-relaxation at fine level
      self.levels[fl].Relax(self.postRelaxationSweeps)


  def MultigridCycle(self):
    self.CycleAtLevel(0)

  def MultigridFCycle(self):
    # smooth and restrict down to coarsest grid
    for level in range(0, self.numLevel - 1):
      self.levels[level].AssignBCs(self.levels[level].solution, True)
      # pre-relaxation at fine level
      self.levels[level].Relax(self.preRelaxationSweeps)

      # coarse grid correction
      cl = level + 1
      self.levels[cl].solution = self.RestrictionSol(level)
      self.levels[cl].forcing = self.levels[cl].Ax() + self.Restriction(level)
    self.FullMultigridCycle()


  def FullMultigridCycle(self):
    # start at coarest grid and obtain solution
    for level in range(self.numLevel - 1, -1, -1):
      self.CycleAtLevel(level)
      # interpolate solution at level to next finest grid
      if level > 0:
        #self.HighOrderInterp2(level)
        #self.InverseDistanceInterp(level)
        nodalSolution = self.levels[level].ToNodes()
        self.Prolongation(level, nodalSolution, False)


