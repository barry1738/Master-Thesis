import torch
import torch.nn as nn
from torch.func import functional_call, vmap, jacrev


def networks_training(model, points_data, rhs_data, epochs, tol, device):
    def compute_loss(model, params, x, z, rhs_f):
        """ Compute the loss function """
        loss = functional_call(model, params, (x, z)) - rhs_f
        return loss
    
    def weights_init(model):
        """Initialize the weights of the neural network."""
        if isinstance(model, nn.Linear):
            # nn.init.xavier_uniform_(model.weight.data, gain=1)
            nn.init.xavier_normal_(model.weight.data, gain=2)

    def qr_decomposition(J_mat, diff, mu):
        """ Solve the linear system using QR decomposition """
        A = torch.vstack((J_mat, mu**0.5 * torch.eye(J_mat.size(1), device=device)))
        b = torch.vstack((-diff, torch.zeros(J_mat.size(1), 1, device=device)))
        Q, R = torch.linalg.qr(A)
        x = torch.linalg.solve_triangular(R, Q.t() @ b, upper=True)
        return x.flatten()

    def cholesky(J, diff, mu):
        """ Solve the linear system using Cholesky decomposition """
        A = J.t() @ J + mu * torch.eye(J.shape[1], device=device)
        b = J.t() @ -diff
        L = torch.linalg.cholesky(A)
        y = torch.linalg.solve_triangular(L, b, upper=False)
        x = torch.linalg.solve_triangular(L.t(), y, upper=True)
        return x.flatten()
    
    # Initialize the weights of the neural network
    model.apply(weights_init)
    params = model.state_dict()

    # Convert the data to torch tensors
    x, z, x_valid, z_valid = points_data
    rhs_f, rhs_f_valid = rhs_data

    x, z = torch.from_numpy(x).to(device), torch.from_numpy(z).to(device)
    x_valid, z_valid = torch.from_numpy(x_valid).to(device), torch.from_numpy(z_valid).to(device)
    rhs_f = torch.from_numpy(rhs_f).to(device)
    rhs_f_valid = torch.from_numpy(rhs_f_valid).to(device)
    print(f"x = {x.dtype}, z = {z.dtype}, rhs_f = {rhs_f.dtype}")
    
    mu = 1.0e3
    savedloss = []
    saveloss_vaild = []

    for step in range(epochs):
        jac_dict = vmap(
            jacrev(compute_loss, argnums=1),
            in_dims=(None, None, 0, 0, 0),
            out_dims=0,
        )(model, params, x, z, rhs_f)

        # Compute the Jacobian matrix
        jac = torch.hstack([v.view(x.size(0), -1) for v in jac_dict.values()])
        jac /= torch.sqrt(torch.tensor(x.size(0)))

        # Compute the residual vector
        l_vec = compute_loss(model, params, x, z, rhs_f)
        l_vec_valid = compute_loss(
            model, params, x_valid, z_valid, rhs_f_valid
        )
        l_vec /= torch.sqrt(torch.tensor(x.size(0)))
        l_vec_valid /= torch.sqrt(torch.tensor(x_valid.size(0)))

        # Solve the linear system
        # p = qr_decomposition(J_mat, L_vec, mu)
        p = cholesky(jac, l_vec, mu)
        params_flatten = nn.utils.parameters_to_vector(params.values())
        params_flatten += p
        # Update the model parameters
        nn.utils.vector_to_parameters(params_flatten, params.values())

        # Compute the loss function
        loss = torch.sum(l_vec**2)
        loss_valid = torch.sum(l_vec_valid**2)
        savedloss.append(loss.item())
        saveloss_vaild.append(loss_valid.item())

        if step % 50 == 0:
            print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

        # Update mu or Stop the iteration
        if loss < tol:
            print("Training successful.")
            print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")
            break
        elif step % 3 == 0:
            if savedloss[step] > savedloss[step - 1]:
                mu = min(mu * 2.0, 1.0e8)
            else:
                mu = max(mu / 1.5, 1.0e-12)

    else:
        print("Reach the maximum number of iterations")
        print(f"step = {step}, loss = {loss.item():.2e}, mu = {mu:.1e}")

    return params, savedloss, saveloss_vaild
