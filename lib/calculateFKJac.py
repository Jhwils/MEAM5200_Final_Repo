import numpy as np
from math import pi
import math

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        self.DH_Param = np.array([[0,  0.141,  0,  0],
                                  [0, 0.192,  0, -pi/2],
                                  [0,   0,    0,  pi/2],
                                  [0, 0.316,  0.0825,  pi/2],
                                  [0,   0,    -0.0825, -pi/2],
                                  [0, 0.384,  0,  pi/2],
                                  [0,   0, 0.088, pi/2],
                                  [0, 0.210,  0,  0]])

    def homogeneousTransform(self, theta, d, a, alpha):

        Rot_z = np.array([[math.cos(theta), -1*math.sin(theta), 0,  0],
                          [math.sin(theta),    math.cos(theta), 0,  0],
                          [     0,                   0,         1,  0],
                          [     0,                   0,         0,  1]])

        Trans_z = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, d],
                            [0, 0, 0, 1]])

        Trans_a = np.array([[1, 0, 0, a],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        Rot_x = np.array([[     1,              0,                0,          0],
                          [     0,      math.cos(alpha), -1*math.sin(alpha),  0],
                          [     0,      math.sin(alpha),    math.cos(alpha),  0],
                          [     0,              0,                0,          1]])

        ht_matrix = Rot_z @ Trans_z @ Trans_a @ Rot_x 

        return ht_matrix

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))

        T = np.identity(4)

        DH_current = self.DH_Param.copy()
        DH_current[1:8, 0] = q
        DH_current[7, 0] += -pi/4 # offset in last frame

        # Calculate the physical joint positions and transformation matrix
        for i in range(0, 8):
            theta, d, a, alpha = DH_current[i]

            T = T @ self.homogeneousTransform(theta, d, a, alpha) # Calculate T(i-1)i

            #add the offset
            if i == 2: #joint 3
                T2 = T @ np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0.195],
                                    [0, 0, 0, 1]])
                jointPositions[i,:] = T2[:3, -1]
                T0e[i,:,:] = T2

            elif i == 4: #joint 5
                T4 = T @np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.125],
                                  [0, 0, 0, 1]])
                jointPositions[i,:] = T4[:3, -1]
                T0e[i,:,:] = T4

            elif i == 5: #joint 6
                T5 = T @np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, -0.015],
                                  [0, 0, 0, 1]])
                jointPositions[i,:] = T5[:3, -1]
                T0e[i,:,:] = T5
            
            elif i == 6: #joint 7
                T6 = T @np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.051],
                                  [0, 0, 0, 1]])
                jointPositions[i,:] = T6[:3, -1]
                T0e[i,:,:] = T6

            else:
                jointPositions[i, :] = T[:3, -1]
                T0e[i,:,:] = T

        # Calculate the virtual joint positions and transformation matrix

        Tee = T # transformation matrix of end effector
        T_virtual_1 = Tee @ np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0.100],
                                      [0, 0, 1, -0.105],
                                      [0, 0, 0, 1]])
        
        T_virtual_2 = Tee @ np.array([[1, 0, 0, 0],
                                      [0, 1, 0, -0.100],
                                      [0, 0, 1, -0.105],
                                      [0, 0, 0, 1]])

        jointPositions[8,:] = T_virtual_1[:3, -1]
        jointPositions[9,:] = T_virtual_2[:3, -1]

        T0e[8,:,:] = T_virtual_1
        T0e[9,:,:] = T_virtual_2

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. 
            Transformations are not necessarily located at the joint locations (standard DH frames).
        """
        # STUDENT CODE HERE
        Ai = []
        T = np.identity(4)

        DH_current = self.DH_Param.copy()
        DH_current[1:8, 0] = q
        DH_current[7, 0] += -pi/4 # offset in last frame

        # Calculate T0i
        for i in range(0, 8):
            theta, d, a, alpha = DH_current[i]

            T_increment = self.homogeneousTransform(theta, d, a, alpha) 
            
            # T0i = T0(i-1) * T(i-1)i
            T = T @ T_increment 
            
            # Save
            Ai.append(T.copy()) 

        Tee = T # transformation matrix of end effector

        T_virtual_1 = Tee @ np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0.100],
                                      [0, 0, 1, -0.105],
                                      [0, 0, 0, 1]])
        
        T_virtual_2 = Tee @ np.array([[1, 0, 0, 0],
                                      [0, 1, 0, -0.100],
                                      [0, 0, 1, -0.105],
                                      [0, 0, 0, 1]])

        Ai.append(T_virtual_1.copy())
        Ai.append(T_virtual_2.copy())

        return Ai
    
    
    def Jv_i(self, q, i): 
        # i: from 0 to 8
        Jvi = np.zeros((3, 9))
        Ai = self.compute_Ai(q)

        T0i = Ai[i]
        p_i = T0i[:3, 3] # position

        loop_range = min(i+1, 7) # whether is the physical joints or not

        # calculate Jv_i
        for j in range(loop_range):
            T0j = Ai[j]
            p_j = T0j[:3, 3] # position
            z_j = T0j[:3, 2] # z axis

            Jv_ij = np.cross(z_j, (p_i - p_j)) # cross product

            Jvi[:,j] = Jv_ij

        return Jvi




if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
