import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# Import from lib
from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.IK_position_null import IK



if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    # Scan position: rotate right (joint 1) and tilt wrist down (joint 6) from start pose
    # Joint 1: -0.3 (rotate right)
    # Joint 6: negative offset to look down
    scan_position = np.array([-0.3, -0.76012354, 0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2 - 1.05, 0.75344866])

    # We need to create a counter to record the desired placing height
    num_stacked = 0

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    fk_solver = FK()
    ik_solver = IK()

    #!!!!!!!!!
    q_in = start_position # this parameter need to change
    #!!!!!!!!!

    while True:
        print(f"Current stacked blocks: {num_stacked}")

        # Move to scan position before detecting
        arm.safe_move_to_position(scan_position)
        q_in = scan_position

        # get the transform from camera to panda_end_effector
        H_ee_camera = detector.get_H_ee_camera()
        print(f"DEBUG H_ee_camera: {H_ee_camera}")

        # Get detections and debug
        detections = detector.get_detections()
        print(f"DEBUG detections: {detections}")

        # $$T^{base}_{block} = T^{base}_{ee} \cdot T^{ee}_{camera} \cdot T^{camera}_{block}$$
        # First get T^world_ee
        _, T_base_ee = fk_solver.forward(q_in)

        # We have H_ee_camera, so we need to get H_camera_block
        H_camera_block = detector.get_detections()

        # Detect some blocks...
        block_positions = []

        # pose is the transformation matrix
        for (name, pose) in detector.get_detections():
            # we should multiply these three matrix to get H_world_block
            H_world_block = T_base_ee @ H_ee_camera @ pose
            block_positions.append((name, H_world_block))
            print(f"Stored coordinate for {name}")
        
        # (1). Handling with static blocks
        static_block = []
        for (name, pose) in block_positions:
            if "static" in name:
                pos_y = pose[1, 3] # Y position in Base Frame

                # Filter logic based on Team Color
                # Red Table is at Y > 0 (+0.26), Source is at Y < 0 (-0.17)
                if team == 'red':
                    if pos_y > 0.1: # Threshold to ignore table blocks
                        print(f"Skipping placed block {name} on Red table.")
                        continue
                
                # Blue Table is at Y < 0 (-0.26), Source is at Y > 0 (+0.17)
                else: 
                    if pos_y < -0.1: # Threshold to ignore table blocks
                        print(f"Skipping placed block {name} on Blue table.")
                        continue
                
                static_block.append((name, pose))

        # If the list is empty, it means all blocks have been moved
        if len(static_block) == 0:
            print("All blocks have been moved!")
            break
        

        closest_distance = 100.0  # firstly set a relatively great val
        target_block_name = None
        target_block_pose = None
        
        # Obtain the position of end effector
        current_ee_pos = T_base_ee[:3, 3]

        # 1. Get the position of block
        for (name, pose) in static_block:
            s_block_position = pose[:3, 3]
            # Calculate the distance between end effector and static blocks
            dist = np.linalg.norm(current_ee_pos - s_block_position)
        
            # 3. compare and update the min val
            if dist < closest_distance:
                closest_distance = dist
                target_block_name = name
                target_block_pose = pose
        print(f"The nearest block is {target_block_name}, distance: {closest_distance:.3f}m")

        # 2. Pre-grasp
        if target_block_pose is not None:
            # Overwrite Z to table height (0.20m) + half block (0.025m) = 0.225m
            target_block_pose[2, 3] = 0.225
            
            # 1. Calculate Yaw angle
            theta = np.arctan2(target_block_pose[1, 0], target_block_pose[0, 0])

            # 2. Construct rotation matrices
            c, s = np.cos(theta), np.sin(theta)
            R_z_yaw = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            # Rotate 180 degrees around X axis -> Z becomes -Z (pointing down), X remains (align Yaw)
            R_x_180 = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])
            
            T_pre_grasp = np.identity(4)
            # Matrix multiplication: first rotate Yaw, then flip Z
            T_pre_grasp[:3, :3] = R_z_yaw @ R_x_180
            T_pre_grasp[:3, 3] = target_block_pose[:3, 3] + np.array([0, 0, 0.15]) 

            T_grasp = deepcopy(T_pre_grasp)
            T_grasp[:3, 3] -= np.array([0, 0, 0.15])

            print("Start IK(Pre-grasp)...")
            q_pre_target, _, success_p, message_p = ik_solver.inverse(T_pre_grasp, q_in, 'J_pseudo', 0.5)
            
            if not success_p:
                print("  [IK WARN] Standard IK failed. Trying 180-degree symmetry rotation...")
                # Destruct uring around Z axis by 180 degrees
                R_symmetry = np.array([
                    [-1, 0, 0, 0],
                    [ 0,-1, 0, 0],
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1]
                ])
                # Let the target matrix rotate 180 degrees
                T_pre_grasp_180 = T_pre_grasp @ R_symmetry
                q_pre_target, _, success_p, message_p = ik_solver.inverse(T_pre_grasp_180, q_in, 'J_pseudo', 0.5)

            if success_p == 1:
                print("Move to pre-grasp gesture")
                arm.safe_move_to_position(q_pre_target)
                q_in = q_pre_target # Update seed

                # Process of grasping block
                # 1. Open the gripper
                arm.exec_gripper_cmd(0.080)

                _, T_actual_pre = fk_solver.forward(q_pre_target)
                T_grasp = deepcopy(T_actual_pre)
                T_grasp[2, 3] -= 0.15

                # 2. Move to grasp position
                q_target, _, success_g, message_g = ik_solver.inverse(T_grasp, q_pre_target, 'J_pseudo', 0.5)
                if success_g == 1:
                    print("Start IK(Grasp)...")
                    arm.safe_move_to_position(q_target)

                    # 3. Close the gripper
                    arm.exec_gripper_cmd(0.03, 50)
                    
                    # Verify grasp success
                    rospy.sleep(0.5)  # waiting for gripper to fully settle
                    gripper_state = arm.get_gripper_state()
                    gripper_width = gripper_state['position'][0] + gripper_state['position'][1]
                    gripper_force = gripper_state['force'][0] + gripper_state['force'][1]

                    # Block (5cm) should stop gripper above target (3cm)
                    # Success: width > 0.04m (block stopped it) OR force detected (squeezing something)
                    if gripper_width < 0.04 and gripper_force < 5.0: # will need to tune these based on tests
                        print(f"Grasp FAILED - no block (width: {gripper_width:.3f}m, force: {gripper_force:.1f}N)")
                        arm.exec_gripper_cmd(0.080)  # Re-open
                        arm.safe_move_to_position(q_pre_target)
                        continue  # move on to next block (or maybe try again a certain # of times?)

                    print(f"Grasp SUCCESS (width: {gripper_width:.3f}m, force: {gripper_force:.1f}N)")

                    # 4. Move back to pre-grasp position
                    print("Return to pre-grasp gesture")
                    arm.safe_move_to_position(q_pre_target)

                    print("Grasping completed.")

                    # Move the block to desired location
                    # The world frame is parallel to the base, +- 0.990m on x axis
                    if team == 'red':
                        # red team: base at -0.990, table at -0.731 -> relative +0.259
                        table_pos_base = np.array([0.562, -0.731 + 0.990, 0.200])
                    else:
                        # Blue team: base at +0.990, table at +0.731 -> relative -0.259
                        table_pos_base = np.array([0.562, 0.731 - 0.990, 0.200])

                    # B. Calculate target z height
                    z_target = 0.200 + (num_stacked * 0.05) + 0.025

                    # C. Build Pre-place matrix (high-altitude approach)
                    T_pre_place = np.eye(4)  # Initialize as identity
                    T_pre_place[[1, 2], [1, 2]] = -1
                    T_pre_place[:3, 3] = table_pos_base # Move to the table xy first
                    T_pre_place[2, 3] = z_target + 0.10 # Height = Stacking height + 10cm safety margin

                    # D. Build Place matrix (release point)
                    T_place = deepcopy(T_pre_place)
                    T_place[2, 3] = z_target + 0.005 # Micro-drop height

                    print(f"--> Moving to Table (Stack {num_stacked})...")
                    q_pre_place, _, success_rp, _ = ik_solver.inverse(T_pre_place, q_in, 'J_pseudo', 0.5)

                    if success_rp:
                        # 1. High-altitude approach
                        arm.safe_move_to_position(q_pre_place)
                        q_in = q_pre_place # Update seed

                        # 2. Vertical descent
                        q_place, _, success_r, _ = ik_solver.inverse(T_place, q_in, 'J_pseudo', 0.5)
                        if success_r:
                            arm.safe_move_to_position(q_place)
                            q_in = q_place # Update seed

                            # 3. Release block
                            arm.exec_gripper_cmd(0.080)

                            # 4. Retreat (return to high altitude)
                            arm.safe_move_to_position(q_pre_place)
                            q_in = q_pre_place # Update seed

                            print(f"Block {num_stacked} placed successfully!")
                            num_stacked += 1
                        else:
                            print("Place Down IK Failed.")
                    else:
                        print("Pre-place IK Failed.")
                else:
                    print(f"Grasping IK failed: {message_g}")
            



    # END STUDENT CODE
