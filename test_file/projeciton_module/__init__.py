import projeciton_module.mesh_generator
import projeciton_module.config
import projeciton_module.utilities

# Some Constants
Re = projeciton_module.config.REYNOLDS_NUM
Dt = projeciton_module.config.TIME_STEP

# Mesh Generators
create_square_mesh = projeciton_module.mesh_generator.CreateSquareMesh
create_circle_mesh = projeciton_module.mesh_generator.CreateCircleMesh

exact_sol = projeciton_module.utilities.exact_sol
qr_decomposition = projeciton_module.utilities.qr_decomposition
cholesky = projeciton_module.utilities.cholesky

# Model
predict = projeciton_module.utilities.predict
predict_dx = projeciton_module.utilities.predict_dx
predict_dy = projeciton_module.utilities.predict_dy
predict_dxx = projeciton_module.utilities.predict_dxx
predict_dxy = projeciton_module.utilities.predict_dxy
predict_dyx = projeciton_module.utilities.predict_dyx
predict_dyy = projeciton_module.utilities.predict_dyy
