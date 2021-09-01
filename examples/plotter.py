import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3d(data):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot(data[0], data[1], data[2], 'b')
  plt.show()

def plot2d(data):
  plt.plot(data[0], data[1], 'b')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()

if __name__=="__main__":
  parser = argparse.ArgumentParser("Plotter for SE2 and SE3 examples")
  parser.add_argument("filename", type=str, help="Path to the file containing the data")
  args = parser.parse_args()

  filename = args.filename

  data = np.loadtxt(filename)

  if data.shape[1] == 3:
    plot3d(data.T)
  else:
    plot2d(data.T)
