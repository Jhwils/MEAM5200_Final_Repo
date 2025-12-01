import numpy as np
from math import pi

# Helper function to build a single DH transformation matrix
def dh_matrix(a, alpha, d, theta):
    """
    Constructs a 4x4 homogeneous transformation matrix from DH parameters.
    
    Args:
        a (float): link length
        alpha (float): link twist (in radians)
        d (float): link offset
        theta (float): joint angle (in radians)
        
    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1]
    ])


class FK():
    """
    A class for calculating the forward kinematics and Jacobian of the Panda robot.
    """
    def __init__(self):
        """
        Initializes the robot with its geometric (DH) parameters.
        """
        # DH parameters for the Panda arm [a, alpha, d, theta_offset]
        self.dh_params = [
            [0,       -pi/2,  0.333,  0],     # Joint 1 to 2
            [0,        pi/2,  0,      0],     # Joint 2 to 3
            [0.0825,   pi/2,  0.316,  0],     # Joint 3 to 4
            [-0.0825, -pi/2,  0,      0],     # Joint 4 to 5
            [0,        pi/2,  0.384,  0],     # Joint 5 to 6
            [0.088,    pi/2,  0,      0],     # Joint 6 to 7
            [0,        0,     0.210,  0]      # Joint 7 to End-Effector Frame
        ]

    def forward(self, q, return_all_transforms=False):
        """
        Calculates the forward kinematics of the robot arm. This is the single
        source of truth for all kinematic calculations.

        Args:
            q (np.ndarray): A 1x7 vector of joint angles [q0, ..., q6].
            return_all_transforms (bool, optional): If True, the method will also
                return the list of intermediate transformation matrices (T_0^i). 
                Defaults to False.

        Returns:
            If return_all_transforms is False:
                jointPositions (np.ndarray): 8x3 matrix of joint center coordinates.
                T0e (np.ndarray): 4x4 homogeneous transformation of the end-effector.
            If return_all_transforms is True:
                jointPositions (np.ndarray): 8x3 matrix of joint center coordinates.
                T0e (np.ndarray): 4x4 homogeneous transformation of the end-effector.
                T_list (list): A list of 7 intermediate 4x4 transformation matrices.
        """

        T_list = []

        T_world_to_current = np.identity(4)

        # Loop through joints to calculate all T0i matrices
        for i in range(7):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = q[i] + theta_offset
            Ai = dh_matrix(a, alpha, d, theta)
            
            # T^World_Current means current coordinate in world coordinate
            T_world_to_current = np.dot(T_world_to_current, Ai)

            # Storing current T0i matrix
            T_list.append(T_world_to_current)

        # Apply the final rotation to get the end-effector pose
        theta_final_rotation = -pi/4
        T_7_to_ee = dh_matrix(0, 0, 0, theta_final_rotation)
        T0e = np.dot(T_world_to_current, T_7_to_ee)

        #Joint position calculation
        jointPositions = np.zeros((8, 3))
        T_base_offset = np.dot(np.identity(4), dh_matrix(0, 0, 0.141, 0))
        jointPositions[0, :] = T_base_offset[:3, 3]

        for i in range(len(T_list)):
            T_world_to_joint_frame = T_list[i]
            if i == 1: # joint 3
                T_offset = np.dot(T_world_to_joint_frame, dh_matrix(0, 0, 0.195, 0))
                jointPositions[i+1, :] = T_offset[:3, 3]
            elif i == 3: # joint 5
                T_offset = np.dot(T_world_to_joint_frame, dh_matrix(0, 0, 0.125, 0))
                jointPositions[i+1, :] = T_offset[:3, 3]
            elif i == 4: # joint 6
                T_offset = np.dot(T_world_to_joint_frame, dh_matrix(0, 0, -0.015, 0))
                jointPositions[i+1, :] = T_offset[:3, 3]
            elif i == 5: # joint 7
                T_offset = np.dot(T_world_to_joint_frame, dh_matrix(0, 0, 0.051, 0))
                jointPositions[i+1, :] = T_offset[:3, 3]
            else:
                jointPositions[i+1, :] = T_world_to_joint_frame[:3, 3]

        # Return Logic
        if return_all_transforms:
            return jointPositions, T0e, T_list
        else:
            return jointPositions, T0e


    def get_Jacobian(self, q):
        """
        Calculates the 6x7 geometric Jacobian matrix.
        This method relies on the forward() method to get all necessary kinematic data.

        Args:
            q (np.ndarray): A 1x7 vector of joint angles [q0, ..., q6].

        Returns:
            np.ndarray: The 6x7 geometric Jacobian matrix.
        """
        # Step 1: Call the verified forward method to get all kinematic data.
        _, T0e, T_list = self.forward(q, return_all_transforms=True) ## Ignore the first output

        # Step 2: Extract the end-effector position vector.
        p_ee = T0e[:3, 3]
        
        # Step 3: Calculate each column of the Jacobian.
        J = np.zeros((6, 7))

        # --- Column 0 (for Joint 1) ---
        # The first joint rotates around the base frame's Z-axis.
        z0 = np.array([0.0, 0.0, 1.0])
        p0 = np.array([0.0, 0.0, 0.0])
        J[:3, 0] = np.cross(z0, p_ee - p0)  # Linear velocity part
        J[3:, 0] = z0                        # Angular velocity part

        # --- Columns 1 through 6 (for Joints 2 through 7) ---
        for i in range(1, 7):
            # For joint i+1, we need the frame information of joint i (T_0^i).
            # T_list is 0-indexed, so T_0^i is at T_list[i-1].
            T_prev = T_list[i-1]
            
            # Position of the current joint's origin (p_{i-1})
            p_prev = T_prev[:3, 3]
            
            # Rotation axis of the current joint (z_{i-1})
            z_prev = T_prev[:3, 2]
            
            # Linear velocity part: J_vi = z_{i-1} x (p_ee - p_{i-1})
            J[:3, i] = np.cross(z_prev, p_ee - p_prev)
            
            # Angular velocity part: J_wi = z_{i-1}
            J[3:, i] = z_prev
            
        return J

    # This code is for Lab 2, you can ignore it for Lab 1
    def get_axis_of_rotation(self, q):
        """
        Placeholder for Lab 2.
        """
        return()
    
    # This code is for Lab 2, you can ignore it for Lab 1
    def compute_Ai(self, q):
        """
        Placeholder for Lab 2.
        """
        return()

if __name__ == "__main__":
    # We need to create an instance to use the class
    fk = FK()

    # A test configuration that matches the handout figure.
    q = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])

    # Set numpy print options for better readability.
    np.set_printoptions(precision=4, suppress=True)

    # --- Test the forward() method ---
    # This call uses the default 'return_all_transforms=False'.
    print("--- Testing forward() ---")
    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n", joint_positions)
    print("\nEnd Effector Pose (T0e):\n", T0e)
    print("-" * 30)

    # --- Test the get_Jacobian() method ---
    # This call will internally use 'return_all_transforms=True'.
    print("\n--- Testing get_Jacobian() ---")
    jacobian = fk.get_Jacobian(q)
    print("Jacobian:\n", jacobian)