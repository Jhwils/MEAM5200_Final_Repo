import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))


    #  2. Find valid indices 
    xi = np.vstack((v_in, omega_in)) # xi is (6, 1)

    # Find all non NaN element indices (this is key!)
    # np.isnan(xi) is (6, 1) boolean, .flatten() makes it (6,)
    valid_indices = np.where(~np.isnan(xi.flatten()))[0]

    #  3. Obtain Jacobian 
    J = calcJacobian(q_in) # J is (6, 7)

    #  4. Filter J and xi based on valid tasks 
    # J_task only contains rows corresponding to the speeds we care about
    J_task = J[valid_indices, :]   # shape is (n, 7), n is the number of valid indices
    xi_task = xi[valid_indices, :] # shape is (n, 1)

    #  5. Compute pseudoinverse of J_task
    J_task_pinv = np.linalg.pinv(J_task) # shape is (7, n)

    #  6. Compute primary task joint velocities 
    # This is equivalent to what IK_velocity(q_in, v_in, omega_in) should do
    dq_ik = (J_task_pinv @ xi_task) # shape is (7, 1)

    #  7. Compute null space projection matrix
    # N = (I - J_task_pinv * J_task)
    N = np.eye(7) - (J_task_pinv @ J_task) # shape is (7, 7)

    #  8. Compute null space velocities ---
    # dq_null = N * b
    dq_null = N @ b # shape is (7, 1)

    return (dq_ik + dq_null).T

