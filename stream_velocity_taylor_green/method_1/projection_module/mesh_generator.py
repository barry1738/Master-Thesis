import torch
import numpy as np
from scipy.stats import qmc


class CreateSquareMesh:
    def inner_points(self, nx):
        points = torch.from_numpy(qmc.LatinHypercube(d=2).random(n=nx))
        points_x = points[:, 0].reshape(-1, 1)
        points_y = points[:, 1].reshape(-1, 1)
        return points_x, points_y

    def boundary_points(self, nx):
        left_bd_x = torch.full((nx, 1), 0.0)
        right_bd_x = torch.full((nx, 1), 1.0)
        top_bd_x = torch.vstack((
            torch.tensor(0.0), torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx-2)), torch.tensor(1.0)))
        bottom_bd_x = torch.vstack((
            torch.tensor(0.0), torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx-2)), torch.tensor(1.0)))

        left_bd_y = torch.vstack((
            torch.tensor(0.0), torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx-2)), torch.tensor(1.0)))
        right_bd_y = torch.vstack((
            torch.tensor(0.0), torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx-2)), torch.tensor(1.0)))
        top_bd_y = torch.full((nx, 1), 1.0)
        bottom_bd_y = torch.full((nx, 1), 0.0)

        points_x = torch.vstack((left_bd_x, right_bd_x, top_bd_x, bottom_bd_x))
        points_y = torch.vstack((left_bd_y, right_bd_y, top_bd_y, bottom_bd_y))
        return points_x, points_y
    
    def normal_vector(self, x, y):
        nx_left = torch.full((x.size(0)//4, 1), -1.0)
        nx_right = torch.full((x.size(0)//4, 1), 1.0)
        nx_top = torch.zeros((x.size(0)//4, 1))
        nx_bottom = torch.zeros((x.size(0)//4, 1))

        ny_left = torch.zeros((y.size(0)//4, 1))
        ny_right = torch.zeros((y.size(0)//4, 1))
        ny_top = torch.full((y.size(0)//4, 1), 1.0)
        ny_bottom = torch.full((y.size(0)//4, 1), -1.0)

        nx = torch.vstack((nx_left, nx_right, nx_top, nx_bottom))
        ny = torch.vstack((ny_left, ny_right, ny_top, ny_bottom))
        return nx, ny


class CreateCircleMesh:
    def __init__(self, *, xc=0, yc=0, r=1):
        self.xc = xc
        self.yc = yc
        self.r = r

    def inner_points(self, nx):
        radius = torch.tensor(self.r * np.sqrt(qmc.LatinHypercube(d=1).random(n=nx)))
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=nx))
        points_x = self.xc + radius * torch.cos(theta)
        points_y = self.yc + radius * torch.sin(theta)
        return points_x, points_y

    def boundary_points(self, nx):
        theta = torch.tensor(2 * np.pi * qmc.LatinHypercube(d=1).random(n=nx))
        points_x = self.xc + self.r * torch.cos(theta)
        points_y = self.yc + self.r * torch.sin(theta)
        return points_x, points_y

    def normal_vector(self, x, y):
        nx = 2 * x / 1
        ny = 2 * y / 1
        dist = torch.sqrt(nx**2 + ny**2)
        nx = nx / dist
        ny = ny / dist
        return nx, ny