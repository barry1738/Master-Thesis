import projection_module.mesh_generator
import projection_module.config
import projection_module.utilities
import projection_module.prediction_rhs
import projection_module.prediction_step
import projection_module.projection_step
import projection_module.update_step

# Some Constants
REYNOLDS_NUM = projection_module.config.REYNOLDS_NUM
TIME_STEP = projection_module.config.TIME_STEP

# Mesh Generators
CreateSquareMesh = projection_module.mesh_generator.CreateSquareMesh
CreateCircleMesh = projection_module.mesh_generator.CreateCircleMesh

exact_sol = projection_module.utilities.exact_sol
qr_decomposition = projection_module.utilities.qr_decomposition
cholesky = projection_module.utilities.cholesky

prediction_rhs = projection_module.prediction_rhs.prediction_rhs
prediction_step = projection_module.prediction_step.prediction_step
projection_step = projection_module.projection_step.projection_step
update_step = projection_module.update_step.update_step