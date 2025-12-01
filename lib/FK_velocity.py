import numpy as np 
from lib.calcJacobian import calcJacobian

def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.    
    """

    ## STUDENT CODE GOES HERE

    velocity = np.zeros((6, 1))

    # Compute the Velocity by equation: velocity = J * dq
    J = calcJacobian(q_in)  # Get the Jacobian matrix at the given
    velocity = np.dot(J, dq) 

    return velocity

if __name__ == "__main__":

    # A test configuration that matches the handout figure.
    qin = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    dqt = np.array([1,0,0,0,0,0,0])
    vt = FK_velocity(qin,dqt)


    # Set numpy print options for better readability.
    np.set_printoptions(precision=4, suppress=True)

    print("--- Testing FK_velocity() ---")
    print("velocity:\n", vt)    




