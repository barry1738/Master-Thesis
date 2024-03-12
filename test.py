import numpy as np
import matplotlib.pyplot as plt


# Uniform random distribution within a circle
def random_circle(n, xc=0, yc=0, r=1):
    radius = r * np.sqrt(np.random.rand(n))
    theta = 2 * np.pi * np.random.rand(n)
    x = xc + radius * np.cos(theta)
    y = yc + radius * np.sin(theta)
    return x, y


# Uniform random distribution within a polar curve
def random_polar(n):
    theta = 2 * np.pi * np.random.rand(n)
    radius = 1 / 2 + np.sin(5 * theta) / 7
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


# Check the points inside the polar curve, return z = -1 if inside, z = 1 if outside
def check_polar(x, y):
    dist = np.sqrt(x ** 2 + y ** 2) - (1 / 2 + np.sin(5 * np.arctan2(y, x)) / 7)
    z = np.where(dist > 0, 1, -1)
    return z


def main():
    n = 1000
    x, y = random_circle(n)
    x_p, y_p = random_polar(n)
    z = check_polar(x, y)
    sca = plt.scatter(x, y, c=z, s=1)
    plt.scatter(x_p, y_p, s=1)
    plt.colorbar(sca)
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()