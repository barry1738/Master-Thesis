import projeciton_module.mesh_generator
import projeciton_module.config
import projeciton_module.utilities
import projeciton_module.prediction_rhs
import projeciton_module.prediction_step
import projeciton_module.projection_step
import projeciton_module.update_step

# Some Constants
REYNOLDS_NUM = projeciton_module.config.REYNOLDS_NUM
TIME_STEP = projeciton_module.config.TIME_STEP

# Mesh Generators
CreateSquareMesh = projeciton_module.mesh_generator.CreateSquareMesh
CreateCircleMesh = projeciton_module.mesh_generator.CreateCircleMesh

exact_sol = projeciton_module.utilities.exact_sol
qr_decomposition = projeciton_module.utilities.qr_decomposition
cholesky = projeciton_module.utilities.cholesky

prediction_rhs = projeciton_module.prediction_rhs.prediction_rhs
prediction_step = projeciton_module.prediction_step.prediction_step
projection_step = projeciton_module.projection_step.projection_step
update_step = projeciton_module.update_step.update_step