import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    start = start.flatten()
    goal = goal.flatten()

    # initialize path
    path = []

    if isRobotCollided(start, map) or isRobotCollided(goal, map):
        print("Error: Start or Goal configuration is in collision!")
        return np.array(path)

    # We should store two lists: one for nodes and one for parents
    q_list = []
    parent_list = []

    # add start node
    q_list.append(start)
    parent_list.append(-1)  # start has no parent

    max_iter = 10000
    k = 0
    stp_size = 0.5

    while k < max_iter:
        k += 1
        if k % 1000 == 0:
            print(f"RRT Iteration: {k}")

        if random.random() < 0.2:
            # With some probability, sample the goal directly to bias the search
            q_random_sample = goal
        else:
            # Sample random configuration
            q_random_sample = np.random.uniform(low=lowerLim, high=upperLim)

        # Find nearest node in the tree
        # dists = [np.linalg.norm(q - q_random_sample) for q in q_list]
        Q_matrix = np.array(q_list)
        dists = np.linalg.norm(Q_matrix - q_random_sample, axis=1)
        nearest_index = np.argmin(dists)
        q_nearest = q_list[nearest_index]

        # Steer towards the random sample
        direction = q_random_sample - q_nearest
        direction /= np.linalg.norm(direction)  # Normalize the direction


        # # Move a step towards the random sample
        # step_size = 0.2
        # q_new = q_nearest + step_size * direction

        # # Check if the new configuration is valid
        # if isPathValid(q_nearest, q_new, map):
        #     q_list.append(q_new)
        #     parent_list.append(nearest_index)

        # # Check if we reached the goal
        # if np.linalg.norm(q_new - goal) < 0.1 and isPathValid(q_new, goal, map):
        #     q_list.append(goal)
        #     parent_list.append(len(q_list) - 2)
        #     break


        # 2. initialize greedy algorithm
        current_q = q_nearest            # current top of the tree
        current_parent_idx = nearest_index # index

        # 3. start loop
        while True:
            # A. try to steer
            remaining_dist = np.linalg.norm(q_random_sample - current_q)
            if remaining_dist <= stp_size:
                q_new = q_random_sample
            else:
                q_new = current_q + stp_size * direction

            # B. check safety
            if isPathValid(current_q, q_new, map):
                # if safe, append node
                q_list.append(q_new)
                
                parent_list.append(current_parent_idx)
                
                # update
                current_q = q_new
                current_parent_idx = len(q_list) - 1 # update parent node

                # C. check if it attain the goal
                if np.linalg.norm(q_new - goal) < 0.1 and isPathValid(q_new, goal, map):
                    # Attain! append goal
                    q_list.append(goal)
                    parent_list.append(current_parent_idx)
                    print("Goal Reached via Connect!")
                    k = max_iter + 1 
                    break 
                
                # D. check
                if np.array_equal(q_new, q_random_sample):
                    break
            else:
                # obstacle
                break


    # Backtrack to find the path
    if len(q_list) > 0 and np.linalg.norm(q_list[-1] - goal) < 0.1:
        path.append(q_list[-1])
        index = parent_list[-1]
        while index != -1:
            path.append(q_list[index])
            index = parent_list[index]
        path.reverse()


    return np.array(path)

def isPathValid(q_start, q_end, map_struct):
    """
    Helper function to check if path is valid
    :param q_start:      1x7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :param q_end:      1x7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :param map_struct:   map struct
    :return:        true if path is valid, false otherwise
    """
    n_checks = 15
    dist_vec = q_end - q_start
    for i in range(n_checks + 1):
        alpha = i / n_checks
        q_check = q_start + alpha * dist_vec
        if isRobotCollided(q_check, map_struct):
            return False
    return True

def isRobotCollided(q_in, map_struct):
    """
    Helper function to check if robot is in collision at configuration q_in
    :param q_in:    1x7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :param map_struct:   map struct
    :return:        true if in collision, false otherwise
    """

    # GET JOINT positions
    jointPositions = FK().forward(q_in)[0]

    linePt1 = jointPositions[:7,:]
    linePt2 = jointPositions[1:8,:]

    # CHECK COLLISIONS BETWEEN EACH LINK AND EACH OBSTACLE
    for i in range(np.shape(map_struct.obstacles)[0]): ## For each obstacle
        box = map_struct.obstacles[i,:]
        if np.any(detectCollision(linePt1, linePt2, box)): ## IF returned true, there is a collision
            return True
    return False

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
