import torch
import numpy as np
from scipy.stats import qmc
from scipy.special import ellipeinc
from scipy.optimize import root
# from utilities import exact_sol
# import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


class CreateSquareMesh:
    def inner_points(self, nx):
        points = torch.from_numpy(qmc.LatinHypercube(d=2).random(n=nx))
        points_x = points[:, 0].reshape(-1, 1)
        points_y = points[:, 1].reshape(-1, 1)
        # points_x = torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx))
        # points_y = torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx))
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
    

class CreateEllipseMesh:
    def __init__(self, *, xc=0, yc=0, a=1, b=1):
        self.xc = xc
        self.yc = yc
        self.a = a
        self.b = b

    def inner_points(self, nx):
        """Generate random points inside the ellipse"""
        radius = torch.tensor(np.sqrt(qmc.LatinHypercube(d=1).random(n=nx)))
        theta = 2 * torch.pi * torch.tensor(qmc.LatinHypercube(d=1).random(n=nx))
        points_x = self.xc + self.a * radius * torch.cos(theta)
        points_y = self.yc + self.b * radius * torch.sin(theta)

        return points_x, points_y
    
    def boundary_points(self, nx):
        def angles_in_ellipse(a, b, nx):
            angle = 2 * np.pi * np.arange(nx) / nx
            m = 1.0 - (b / a) ** 2
            arc_total_length = ellipeinc(2.0 * np.pi, m)
            arc_lengths = arc_total_length / nx
            arcs = np.arange(nx) * arc_lengths
            res = root(lambda x: (ellipeinc(x, m) - arcs), angle)
            angles = res.x
            return angles
        
        theta = angles_in_ellipse(self.a, self.b, nx)

        x_ellipse = torch.tensor(self.a * np.sin(theta)).reshape(-1, 1)
        y_ellipse = torch.tensor(self.b * np.cos(theta)).reshape(-1, 1)

        return x_ellipse, y_ellipse


class CreateLshapeMesh:
    def inner_points(self, nx):
        nx_short = nx // 3
        nx_long = nx - nx_short

        x_long = (2.0 - 0.0) * torch.from_numpy(
            qmc.LatinHypercube(d=1).random(n=nx_long)
        ) + 0.0
        y_long = (1.0 - 0.0) * torch.from_numpy(
            qmc.LatinHypercube(d=1).random(n=nx_long)
        ) + 0.0

        x_short = (1.0 - 0.0) * torch.from_numpy(
            qmc.LatinHypercube(d=1).random(n=nx_short)
        ) + 0.0
        y_short = (2.0 - 1.0) * torch.from_numpy(
            qmc.LatinHypercube(d=1).random(n=nx_short)
        ) + 1.0

        points_x = torch.vstack((x_long, x_short))
        points_y = torch.vstack((y_long, y_short))

        return points_x, points_y

    def boundary_points(self, nx):
        left_bd_x = torch.full((nx-1, 1), 0.0)
        right_bd_x = torch.full((nx-1, 1), 2.0)
        top_bd_x = torch.vstack(
            (
                torch.tensor(0.0),
                # torch.tensor(1.0),
                (2.0 - 0.0) * torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx - 3))
                + 0.0,
                torch.tensor(2.0),
            )
        )
        bottom_bd_x = torch.vstack(
            (
                torch.tensor(0.0),
                # torch.tensor(1.0),
                (2.0 - 0.0) * torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx - 3))
                + 0.0,
                torch.tensor(2.0),
            )
        )

        left_bd_y = torch.vstack(
            (
                torch.tensor(0.0),
                # torch.tensor(1.0),
                (2.0 - 0.0) * torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx - 3))
                + 0.0,
                torch.tensor(2.0),
            )
        )
        right_bd_y = torch.vstack(
            (
                torch.tensor(0.0),
                # torch.tensor(1.0),
                (2.0 - 0.0) * torch.from_numpy(qmc.LatinHypercube(d=1).random(n=nx - 3))
                + 0.0,
                torch.tensor(2.0),
            )
        )
        top_bd_y = torch.full((nx-1, 1), 2.0)
        bottom_bd_y = torch.full((nx-1, 1), 0.0)

        right_bd_x = torch.where(right_bd_y <= 1.0, right_bd_x, 1.0)
        top_bd_y = torch.where(top_bd_x <= 1.0, top_bd_y, 1.0)

        right_bd_x = torch.vstack((right_bd_x, torch.tensor(1.0)))
        right_bd_y = torch.vstack((right_bd_y, torch.tensor(1.0)))
        top_bd_x = torch.vstack((top_bd_x, torch.tensor(1.0)))
        top_bd_y = torch.vstack((top_bd_y, torch.tensor(1.0)))

        points_x = torch.vstack((left_bd_x, right_bd_x, top_bd_x, bottom_bd_x))
        points_y = torch.vstack((left_bd_y, right_bd_y, top_bd_y, bottom_bd_y))

        return points_x, points_y
    


# # mesh = CreateLshapeMesh()
# mesh = CreateEllipseMesh(xc=0.0, yc=0.0, a=2.0, b=1.0)
# x_inner, y_inner = mesh.inner_points(10000)
# x_bd, y_bd = mesh.boundary_points(100)

# x = torch.vstack((x_inner, x_bd))
# y = torch.vstack((y_inner, y_bd))

# u = exact_sol(x, y, 0.0, 100.0, "u")
# v = exact_sol(x, y, 0.0, 100.0, "v")
# p = exact_sol(x, y, 0.0, 100.0, "p")

# # fig, ax = plt.subplots()
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# # ax.scatter(x_inner, y_inner, s=1, c='b')
# # ax.scatter(x_bd, y_bd, s=10, c="r")
# # ax.scatter(x, y, u, s=1, c=u, cmap="coolwarm")
# # ax.scatter(x, y, v, s=1, c=v, cmap="coolwarm")
# ax.scatter(x, y, p, s=1, c=p, cmap="coolwarm")
# ax.set_aspect("equal")
# plt.show()