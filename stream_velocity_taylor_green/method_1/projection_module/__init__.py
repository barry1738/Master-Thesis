import projection_module.mesh_generator
import projection_module.utilities
import projection_module.initial_value
import projection_module.rhs_func
import projection_module.prediction_step
import projection_module.projection_step
import projection_module.update_step


# Path: stream_velocity_taylor_green/method_1/projection_module/mesh_generator.py
REYNODELS_NUM = 1000
TIME_STEP = 0.02


CreateSquareMesh = projection_module.mesh_generator.CreateSquareMesh
CreateCircleMesh = projection_module.mesh_generator.CreateCircleMesh

exact_sol = projection_module.utilities.exact_sol
qr_decomposition = projection_module.utilities.qr_decomposition
cholesky = projection_module.utilities.cholesky

initial_value = projection_module.initial_value.initial_value
prediction_step_rhs = projection_module.rhs_func.prediction_step_rhs

prediction_step = projection_module.prediction_step.prediction_step
projection_step = projection_module.projection_step.projection_step
update_step = projection_module.update_step.update_step