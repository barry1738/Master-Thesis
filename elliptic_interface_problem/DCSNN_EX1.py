import torch
import scipy
import numpy
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device = ', device)


def exact_sol(x, z):


class CreateMesh:
    def __init__(self) -> None:
        pass

    def interior_points(self, nx):
        x = 2.0 * scipy.stats.qmc.LatinHypercube(d=2).random(n=nx) - 1.0
        return torch.tensor(x, device=device)

    
    def boundary_points(self, nx):
        left_x = numpy.hstack((
            -1.0 * numpy.ones((nx, 1)),
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0
        ))
        right_x = numpy.hstack((
            numpy.ones((nx, 1)),
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0
        ))
        bottom_x = numpy.hstack((
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            -1.0 * numpy.ones((nx, 1))
        ))
        top_x = numpy.hstack((
            2.0 * scipy.stats.qmc.LatinHypercube(d=1).random(n=nx) - 1.0,
            numpy.ones((nx, 1))
        ))
        x = numpy.vstack((left_x, right_x, bottom_x, top_x))
        return torch.tensor(x, device=device)
    
    def interface_points(self, nx):
        theta = 2.0 * numpy.pi * scipy.stats.qmc.LatinHypercube(d=1).random(n=4 * nx)
        x = numpy.hstack((
            0.2 * numpy.cos(theta),
            0.5 * numpy.sin(theta)
        ))
        return torch.tensor(x, device=device)
    
    def sign_x(self, x, y):
        dist = torch.sqrt((x/0.2)**2 + (y/0.5)**2)
        z = torch.where(dist < 1.0, -1.0, 1.0)
        return z
    
    def normal_vector(self, x, y):
        """
        Coompute the normal vector of interface points,
        only defined on the interface
        """
        n_x = 2.0 * x / (0.2**2)
        n_y = 2.0 * y / (0.5**2)
        length = torch.sqrt(n_x**2 + n_y**2)
        normal_x = n_x / length
        normal_y = n_y / length
        return torch.hstack((normal_x, normal_y))


def main():
    mesh = CreateMesh()
    inner_x = mesh.interior_points(10000)
    boundary_x = mesh.boundary_points(1000)
    interface_x = mesh.interface_points(1000)
    print(f'inner_x = {inner_x.shape}')
    print(f'boundary_x = {boundary_x.shape}')
    x, y = interface_x[:, 0].view(-1, 1), interface_x[:, 1].view(-1, 1)
    x_bd, y_bd = boundary_x[:, 0].view(-1, 1), boundary_x[:, 1].view(-1, 1)
    x_if, y_if = interface_x[:, 0].view(-1, 1), interface_x[:, 1].view(-1, 1)

    sign_z = mesh.sign_x(inner_x)
    print(f'sign_z = {sign_z.shape}')
    normal_interface = mesh.normal_vector(interface_x)
    print(f'normal_interface = {normal_interface.shape}')

    fig, ax = plt.subplots()
    ax.scatter(inner_x[:, 0].cpu(), inner_x[:, 1].cpu(), c=sign_z.cpu(), marker='.')
    ax.scatter(boundary_x[:, 0].cpu(), boundary_x[:, 1].cpu(), c='r', marker='.')
    ax.scatter(interface_x[:, 0].cpu(), interface_x[:, 1].cpu(), c='g', marker='.')
    ax.axis('square')
    plt.show()


if __name__ == "__main__":
    main()