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

    # Scan position: end effector pointing straight down, centered above static blocks
    # Computed via IK: X=0.5m, Y=∓0.17m, Z=0.45m
    if team == 'red':
        scan_position = np.array([-0.10150458, -0.11277801, -0.2136254, -1.88260053, -0.02418632, 1.77260987, 0.47638152])
    else:
        scan_position = np.array([-0.04474656555445834, -0.11761014919615738, 0.35110533582159, -1.88248241588739, 0.04099041919237736, 1.7720677504517122, 1.0813846631418398])

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

    q_in = start_position

    # Move to scan position before detecting (SINGLE SCAN)
    arm.safe_move_to_position(scan_position)
    q_in = scan_position

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()

    # Get T^base_ee for coordinate transformation
    _, T_base_ee = fk_solver.forward(q_in)

    # Detect and store all static blocks in base frame (single scan)
    static_blocks_to_pick = []
    for (name, pose) in detector.get_detections():
        if "static" in name:
            # T^base_block = T^base_ee * T^ee_camera * T^camera_block
            H_world_block = T_base_ee @ H_ee_camera @ pose
            pos_y = H_world_block[1, 3]

            # Filter logic based on Team Color
            if team == 'red':
                if pos_y > 0.1:
                    print(f"Skipping placed block {name} on Red table.")
                    continue
            else:
                if pos_y < -0.1:
                    print(f"Skipping placed block {name} on Blue table.")
                    continue

            static_blocks_to_pick.append((name, H_world_block))
            print(f"Stored coordinate for {name}")

    print(f"\nTotal static blocks to pick: {len(static_blocks_to_pick)}\n")

    # Main loop: pick each stored block
    while len(static_blocks_to_pick) > 0:
        print(f"Current stacked blocks: {num_stacked}")

        # Sort blocks by |Y| to pick inner blocks first (closer to origin)
        # This prevents gripper from hitting outer blocks when reaching inward
        static_blocks_to_pick.sort(key=lambda x: abs(x[1][1, 3]))

        # Pop the innermost block (smallest |Y|)
        target_block_name, target_block_pose = static_blocks_to_pick.pop(0)
        print(f"Picking {target_block_name}, Y: {target_block_pose[1,3]:.3f}m")

        # 2. Pre-grasp
        if target_block_pose is not None:
            # Overwrite Z to table height (0.20m) + half block (0.025m) = 0.225m
            target_block_pose[2, 3] = 0.225

            # Get yaw angle directly from block orientation (same as working file)
            theta = np.arctan2(target_block_pose[1, 0], target_block_pose[0, 0])
            print(f"  Block yaw: {np.degrees(theta):.1f}°")

            # Rotation matrix for pointing down (same as working file)
            R_x_180 = np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]
            ])

            # Try 4 orientations: 0°, 90°, 180°, 270° from detected yaw
            path_found = False
            for i, angle_offset in enumerate([0, np.pi/2, np.pi, 3*np.pi/2]):
                test_theta = theta + angle_offset
                c, s = np.cos(test_theta), np.sin(test_theta)
                R_z_yaw = np.array([
                    [c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]
                ])

                T_pre_grasp = np.identity(4)
                T_pre_grasp[:3, :3] = R_z_yaw @ R_x_180
                T_pre_grasp[:3, 3] = target_block_pose[:3, 3] + np.array([0, 0, 0.08])

                T_grasp = deepcopy(T_pre_grasp)
                T_grasp[:3, 3] -= np.array([0, 0, 0.08])

                q_pre, _, ok_pre, _ = ik_solver.inverse(T_pre_grasp, start_position, 'J_pseudo', 0.5)
                if ok_pre:
                    q_grasp, _, ok_grasp, _ = ik_solver.inverse(T_grasp, q_pre, 'J_pseudo', 0.5)
                    if ok_grasp:
                        print(f"  Valid path found (edge {i}, yaw={np.degrees(test_theta):.1f}°)")
                        q_pre_target = q_pre
                        q_grasp_target = q_grasp
                        path_found = True
                        break

            if path_found:
                # 1. Open gripper and move to pre-grasp
                arm.exec_gripper_cmd(0.080)
                print("Move to pre-grasp gesture")
                arm.safe_move_to_position(q_pre_target)
                q_in = q_pre_target

                # 2. Move to grasp position
                print("Move to grasp...")
                arm.safe_move_to_position(q_grasp_target)

                # 3. Close the gripper
                arm.exec_gripper_cmd(0.03, 50)

                # Move the block to desired location
                # Place on goal platform edge closest to static blocks
                if team == 'red':
                    table_pos_base = np.array([0.562, 0.075, 0.200])
                else:
                    table_pos_base = np.array([0.562, -0.075, 0.200])

                # Calculate target z height
                z_target = 0.200 + (num_stacked * 0.05) + 0.025

                # Build Pre-place matrix (high-altitude approach)
                T_pre_place = np.eye(4)
                T_pre_place[[1, 2], [1, 2]] = -1
                T_pre_place[:3, 3] = table_pos_base
                T_pre_place[2, 3] = z_target + 0.06

                # Build Place matrix (release point)
                T_place = deepcopy(T_pre_place)
                T_place[2, 3] = z_target + 0.005

                print(f"--> Moving to Table (Stack {num_stacked})...")
                q_pre_place, _, success_rp, _ = ik_solver.inverse(T_pre_place, q_in, 'J_pseudo', 0.5)

                if success_rp:
                    arm.safe_move_to_position(q_pre_place)
                    q_in = q_pre_place

                    q_place, _, success_r, _ = ik_solver.inverse(T_place, q_in, 'J_pseudo', 0.5)
                    if success_r:
                        arm.safe_move_to_position(q_place)
                        q_in = q_place

                        arm.exec_gripper_cmd(0.080)

                        arm.safe_move_to_position(q_pre_place)
                        q_in = q_pre_place

                        print(f"Block {num_stacked} placed successfully!")
                        num_stacked += 1
                    else:
                        print("Place Down IK Failed.")
                else:
                    print("Pre-place IK Failed.")
            else:
                print(f"  No valid grasp path found for {target_block_name}. Skipping.")
            



    # END STUDENT CODE
