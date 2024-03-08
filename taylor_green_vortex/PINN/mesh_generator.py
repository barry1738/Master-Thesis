import torch
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)


class CreateMesh:
    def create_mesh(self, left, right, top, bottom, domain):
        inside_points = torch.from_numpy(qmc.LatinHypercube(d=2).random(n=domain))

        left_boundary = torch.hstack((
            torch.full((left, 1), 0.0),
            torch.from_numpy(qmc.LatinHypercube(d=1).random(n=left))
        ))
        right_boundary = torch.hstack((
            torch.full((right, 1), 1.0),
            torch.from_numpy(qmc.LatinHypercube(d=1).random(n=right))
        ))
        top_boundary = torch.hstack((
            torch.from_numpy(qmc.LatinHypercube(d=1).random(n=top)),
            torch.full((top, 1), 1.0)
        ))
        bottom_boundary = torch.hstack((
            torch.from_numpy(qmc.LatinHypercube(d=1).random(n=bottom)),
            torch.full((bottom, 1), 0.0)
        ))

        return inside_points, left_boundary, right_boundary, top_boundary, bottom_boundary


# mesh = CreateMesh()
# inside, left, right, top, bottom = mesh.create_mesh(10, 10, 10, 10, 100)
# print(inside.shape)
# print(left.shape)
# print(right.shape)
# print(top.shape)
# print(bottom.shape)