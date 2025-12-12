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

    OFFSET_X = 0.00  
    OFFSET_Y = 0.00
    
    # Extra correction for placing (if the stacked tower is leaning, the place position might need adjustment)
    PLACE_OFFSET_X = 0.00
    PLACE_OFFSET_Y = 0.00

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

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

    # Scan position: end effector pointing straight down, centered above static blocks
    # Computed via IK: X=0.5m, Y=∓0.17m, Z=0.45m
    if team == 'red':
        scan_position = np.array([-0.10150458, -0.11277801, -0.2136254, -1.88260053, -0.02418632, 1.77260987, 0.47638152])
    else:
        scan_position = np.array([-0.04474656555445834, -0.11761014919615738, 0.35110533582159, -1.88248241588739, 0.04099041919237736, 1.7720677504517122, 1.0813846631418398])

    # We need to create a counter to record the desired placing height
    num_stacked = 0

    q_in = start_position

    # Move to scan position before detecting
    arm.safe_move_to_position(scan_position)
    q_in = scan_position

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()

    # $$T^{base}_{block} = T^{base}_{ee} \cdot T^{ee}_{camera} \cdot T^{camera}_{block}$$
    # First get T^world_ee
    _, T_base_ee = fk_solver.forward(q_in)

    # Detect and store all static blocks in base frame (single scan)
    static_blocks_to_pick = []
    for (name, pose) in detector.get_detections():
        if "static" in name:
            # we should multiply these three matrix to get H_world_block
            H_world_block = T_base_ee @ H_ee_camera @ pose
            pos_y = H_world_block[1, 3] # Y position in Base Frame

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

            static_blocks_to_pick.append((name, H_world_block))
            print(f"Stored coordinate for {name}")

    print(f"\nTotal static blocks to pick: {len(static_blocks_to_pick)}\n")

    # TESTING: Uncomment the line below to skip static blocks and test dynamic only
    # static_blocks_to_pick = []

    # If the list is empty, it means all blocks have been moved
    if len(static_blocks_to_pick) == 0:
        print("No static blocks found to pick!")

    # Main loop: pick each stored block
    while len(static_blocks_to_pick) > 0:
        print(f"Current stacked blocks: {num_stacked}")

        static_blocks_to_pick.sort(key=lambda x: abs(x[1][1, 3]))
        target_block_name, target_block_pose = static_blocks_to_pick.pop(0)
        print(f"Picking {target_block_name}...")

        # ---------------------------------------------------------
        # 1. Move to Coarse Pre-grasp (High Altitude)
        # ---------------------------------------------------------
        target_block_pose[2, 3] = 0.225
        target_block_pose[0, 3] += OFFSET_X
        target_block_pose[1, 3] += OFFSET_Y

        theta = np.arctan2(target_block_pose[1, 0], target_block_pose[0, 0])
        c, s = np.cos(theta), np.sin(theta)
        R_z_yaw = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        R_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        T_pre_grasp = np.identity(4)
        T_pre_grasp[:3, :3] = R_z_yaw @ R_x_180
        T_pre_grasp[:3, 3] = target_block_pose[:3, 3] + np.array([0, 0, 0.10]) # +10cm

        print(f"Moving to Coarse Pre-grasp...")
        q_pre_target, _, success_p, _ = ik_solver.inverse(T_pre_grasp, q_in, 'J_pseudo', 0.5)
        
        if not success_p:
            R_sym = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            q_pre_target, _, success_p, _ = ik_solver.inverse(T_pre_grasp @ R_sym, q_in, 'J_pseudo', 0.5)

        if not success_p:
            print("Pre-grasp IK failed. Skipping block.")
            continue

        arm.exec_gripper_cmd(0.080)
        arm.safe_move_to_position(q_pre_target)
        q_in = q_pre_target
        
        print("  Visual Servoing...")
        rospy.sleep(0.5) # Wait for settle
        
        H_ee_camera = detector.get_H_ee_camera()
        _, T_base_ee = fk_solver.forward(q_in)
        detections = detector.get_detections()
        
        refined_pose = None
        min_dist = 100.0
        expected_pos = target_block_pose[:3, 3]

        for (name, pose) in detections:
            H_world_new = T_base_ee @ H_ee_camera @ pose
            dist = np.linalg.norm(H_world_new[:3, 3] - expected_pos)
            if dist < 0.10: 
                if dist < min_dist:
                    min_dist = dist
                    refined_pose = H_world_new
        
        if refined_pose is not None:
            print(f"  Correction: {np.linalg.norm(refined_pose[:3,3] - expected_pos)*1000:.1f} mm")
            
            target_pos_final = refined_pose[:3, 3]
            target_pos_final[2] = 0.225
            target_pos_final[0] += OFFSET_X
            target_pos_final[1] += OFFSET_Y

            theta_refined = np.arctan2(refined_pose[1, 0], refined_pose[0, 0])
            c, s = np.cos(theta_refined), np.sin(theta_refined)
            R_z_yaw = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
            
            # Instead of diving diagonally, we first ALIGN horizontally
            
            # 1. Define Refined Pre-grasp (High Z)
            T_pre_grasp_refined = np.identity(4)
            T_pre_grasp_refined[:3, :3] = R_z_yaw @ R_x_180
            T_pre_grasp_refined[:3, 3] = target_pos_final + np.array([0, 0, 0.10]) # High
            
            # 2. Define Refined Grasp (Low Z)
            T_grasp_final = deepcopy(T_pre_grasp_refined)
            T_grasp_final[:3, 3] -= np.array([0, 0, 0.10]) # Low

            # 3. Move to Refined Pre-grasp (The Shimmy)
            # This ensures we are perfectly above the block before descending
            q_pre_refined, _, success_ref, _ = ik_solver.inverse(T_pre_grasp_refined, q_in, 'J_pseudo', 0.5)
            
            if success_ref:
                arm.safe_move_to_position(q_pre_refined)
                q_in = q_pre_refined
            else:
                # Try symmetry
                R_sym = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
                q_pre_refined, _, success_ref, _ = ik_solver.inverse(T_pre_grasp_refined @ R_sym, q_in, 'J_pseudo', 0.5)
                if success_ref:
                    arm.safe_move_to_position(q_pre_refined)
                    q_in = q_pre_refined
                    # Update grasp matrix to match symmetry
                    T_grasp_final = T_grasp_final @ R_sym
            
        else:
            print("  [WARN] Block lost. Using coarse estimation.")
            T_grasp_final = deepcopy(T_pre_grasp)
            T_grasp_final[:3, 3] -= np.array([0, 0, 0.10])
            T_grasp_final[2, 3] = 0.225

        # Execute Vertical Descent
        print("  Descending to Grasp...")
        q_grasp, _, success_g, _ = ik_solver.inverse(T_grasp_final, q_in, 'J_pseudo', 0.5)
        
        if success_g:
            arm.safe_move_to_position(q_grasp)
            arm.exec_gripper_cmd(0.03, 50)
            
            # Place Logic
            if team == 'red':
                table_pos_base = np.array([0.562, 0.075, 0.200])
            else:
                table_pos_base = np.array([0.562, -0.075, 0.200])

            table_pos_base[0] += PLACE_OFFSET_X
            table_pos_base[1] += PLACE_OFFSET_Y

            z_target = 0.200 + (num_stacked * 0.05) + 0.025

            T_pre_place = np.eye(4)
            T_pre_place[[1, 2], [1, 2]] = -1
            T_pre_place[:3, 3] = table_pos_base
            T_pre_place[2, 3] = z_target + 0.06

            T_place = deepcopy(T_pre_place)
            T_place[2, 3] = z_target + 0.005
            
            print(f"--> Placing on Stack {num_stacked}...")
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
                    num_stacked += 1
                    print("Placed!")
                else:
                    print("Place Down failed")
            else:
                print("Pre-place failed")

        else:
            print("Grasp IK Failed")

    print(f"\n=== Static blocks done: Stacked {num_stacked} blocks ===")


    # DYNAMIC BLOCK HANDLING (Wait-to-Intercept)
    print("\n>>> DYNAMIC MODE (Wait-to-Intercept) <<<\n")

    if team == 'red':
        observation_position = np.array([-0.66902174, -0.80879801,  1.61876292, -1.13930242,  0.84472293,  1.31064934,  2.04106599])
        TURNTABLE_CENTER = np.array([0.0, 0.990])
    else:
        observation_position = np.array([0.67380074, -0.73944342, -1.71271262, -1.14979493, -0.75195756,  1.35414952, -0.49852892])
        TURNTABLE_CENTER = np.array([0.0, -0.990])

    TURNTABLE_HEIGHT = 0.200
    BLOCK_HALF_HEIGHT = 0.025

    print("Moving to observation position...")
    arm.safe_move_to_position(observation_position)
    q_in = observation_position

    H_ee_camera = detector.get_H_ee_camera()

    MAX_DYNAMIC_GRABS = 4
    dynamic_grabbed = 0
    attempt = 0

    while dynamic_grabbed < MAX_DYNAMIC_GRABS:
        attempt += 1
        print(f"\n--- Dynamic Attempt {attempt} ---")

        # 1. Scan
        _, T_base_ee = fk_solver.forward(q_in)
        detections = detector.get_detections()
        t_now = time_in_seconds()

        target_samples = []
        found_block = False

        for (name, pose) in detections:
            if "dynamic" in name:
                H_world_block = T_base_ee @ H_ee_camera @ pose
                x = H_world_block[0, 3]
                y = H_world_block[1, 3]
                theta_block = np.arctan2(H_world_block[1, 0], H_world_block[0, 0])
                target_samples.append((t_now, x, y, theta_block, name))
                found_block = True
                break

        if not found_block:
            print("No dynamic block. Retrying...")
            rospy.sleep(0.5)
            continue

        # 2. Tracking
        NUM_SAMPLES = 12 
        for i in range(NUM_SAMPLES):
            _, T_base_ee = fk_solver.forward(q_in)
            detections = detector.get_detections()
            t_now = time_in_seconds()

            for (name, pose) in detections:
                if "dynamic" in name:
                    H_world_block = T_base_ee @ H_ee_camera @ pose
                    x = H_world_block[0, 3]
                    y = H_world_block[1, 3]
                    theta_block = np.arctan2(H_world_block[1, 0], H_world_block[0, 0])
                    target_samples.append((t_now, x, y, theta_block, name))
                    break
            rospy.sleep(0.05)

        if len(target_samples) < 5:
            continue

        # 3. Estimate Omega
        phis = []
        times = []
        for (t, x, y, theta_b, name) in target_samples:
            phi = np.arctan2(y - TURNTABLE_CENTER[1], x - TURNTABLE_CENTER[0])
            phis.append(phi)
            times.append(t)

        phis = np.unwrap(phis)
        times = np.array(times)

        if len(times) > 1:
            A = np.vstack([times, np.ones(len(times))]).T
            omega, phi0 = np.linalg.lstsq(A, phis, rcond=None)[0]
        else:
            continue

        last_sample = target_samples[-1]
        t_last, x_last, y_last, theta_last, block_name = last_sample
        radius = np.sqrt((x_last - TURNTABLE_CENTER[0])**2 + (y_last - TURNTABLE_CENTER[1])**2)

        print(f"  Omega: {omega:.3f} rad/s")

        # 4. Predict Intercept
        INTERCEPT_DELAY = 3.0 
        t_intercept = time_in_seconds() + INTERCEPT_DELAY
        phi_intercept = omega * t_intercept + phi0
        
        if team == 'red':
            target_phi_ideal = -pi/2
        else:
            target_phi_ideal = pi/2
            
        current_phi = omega * time_in_seconds() + phi0
        delta_phi = target_phi_ideal - current_phi
        while delta_phi / omega < 0:
             if omega > 0: delta_phi += 2*pi
             else: delta_phi -= 2*pi
             
        time_to_travel = delta_phi / omega
        while time_to_travel < 2.0:
            time_to_travel += abs(2*pi / omega)
            
        t_intercept = time_in_seconds() + time_to_travel
        phi_intercept = omega * t_intercept + phi0
        
        x_predict = TURNTABLE_CENTER[0] + radius * np.cos(phi_intercept)
        y_predict = TURNTABLE_CENTER[1] + radius * np.sin(phi_intercept)
        z_predict = TURNTABLE_HEIGHT + BLOCK_HALF_HEIGHT
        
        print(f"  Intercept in {time_to_travel:.2f}s")

        # 5. Compute Pre-Grasp
        theta_intercept = theta_last + omega * (t_intercept - t_last)
        c, s = np.cos(theta_intercept), np.sin(theta_intercept)
        R_z_yaw = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        R_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        T_pre_grasp_dyn = np.identity(4)
        T_pre_grasp_dyn[:3, :3] = R_z_yaw @ R_x_180
        T_pre_grasp_dyn[:3, 3] = np.array([x_predict, y_predict, z_predict + 0.05])

        T_grasp_dyn = deepcopy(T_pre_grasp_dyn)
        T_grasp_dyn[2, 3] = z_predict

        # 6. Move to Pre-Grasp
        print("  Moving to Intercept Point...")
        q_pre_dyn, _, success_p, _ = ik_solver.inverse(T_pre_grasp_dyn, q_in, 'J_pseudo', 0.5)
        if not success_p:
            R_sym = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            q_pre_dyn, _, success_p, _ = ik_solver.inverse(T_pre_grasp_dyn @ R_sym, q_in, 'J_pseudo', 0.5)

        if not success_p:
            print("  Unreachable. Skip.")
            continue

        arm.exec_gripper_cmd(0.080)
        arm.safe_move_to_position(q_pre_dyn)
        q_in = q_pre_dyn
        
        # 7. Wait
        TIME_OFFSET = 0.15 
        while time_in_seconds() < (t_intercept - TIME_OFFSET):
            pass
            
        # 8. Blind Grasp
        print("  GRAB!")
        q_grasp_dyn, _, success_g, _ = ik_solver.inverse(T_grasp_dyn, q_in, 'J_pseudo', 0.5)
        
        if success_g:
            arm.safe_move_to_position(q_grasp_dyn)
            arm.exec_gripper_cmd(0.03, 50)
            
            gripper_state = arm.get_gripper_state()
            if gripper_state['position'][0] < 0.01:
                print("  Missed.")
                arm.exec_gripper_cmd(0.080)
                arm.safe_move_to_position(q_pre_dyn)
                continue
                
            print("  Got it!")
            
            # 9. Place
            if team == 'red':
                table_pos_base = np.array([0.562, 0.075, 0.200])
            else:
                table_pos_base = np.array([0.562, -0.075, 0.200])

            table_pos_base[0] += PLACE_OFFSET_X
            table_pos_base[1] += PLACE_OFFSET_Y

            z_target = 0.200 + (num_stacked * 0.05) + 0.025
            
            T_pre_place = np.eye(4)
            T_pre_place[[1, 2], [1, 2]] = -1
            T_pre_place[:3, 3] = table_pos_base
            T_pre_place[2, 3] = z_target + 0.06
            T_place = deepcopy(T_pre_place)
            T_place[2, 3] = z_target + 0.005

            place_seeds = [q_in, scan_position, start_position]
            success_rp = False
            for seed in place_seeds:
                q_pre_place, _, success_rp, _ = ik_solver.inverse(T_pre_place, seed, 'J_pseudo', 0.5)
                if success_rp: break

            if success_rp:
                arm.safe_move_to_position(q_pre_place)
                q_in = q_pre_place
                q_place, _, success_r, _ = ik_solver.inverse(T_place, q_in, 'J_pseudo', 0.5)
                if success_r:
                    arm.safe_move_to_position(q_place)
                    arm.exec_gripper_cmd(0.080)
                    arm.safe_move_to_position(q_pre_place)
                    q_in = q_pre_place
                    num_stacked += 1
                    dynamic_grabbed += 1
                    print(f"  Placed dynamic block {num_stacked}!")

        arm.safe_move_to_position(observation_position)
        q_in = observation_position

    print(f"\n=== FINAL: Stacked {num_stacked} total blocks ===")
        
    # ==================== DYNAMIC BLOCK HANDLING ====================
    print("\n>>> Switching to DYNAMIC block mode <<<\n")

    # Observation positions over turntable (from DynamicPart.py)
    if team == 'red':
        observation_position = np.array([
            -0.66902174, -0.80879801,  1.61876292,
            -1.13930242,  0.84472293,  1.31064934,  2.04106599
        ])
    else:
        observation_position = np.array([
             0.67380074, -0.73944342, -1.71271262,
            -1.14979493, -0.75195756,  1.35414952, -0.49852892
        ])

    # Turntable parameters (from PDF)
    # Turntable is at world origin (0, 0), but robot base is at Y = ±0.990m
    # So in BASE frame, turntable center is at:
    #   Red team (base at Y = -0.990 in world): turntable at Y = +0.990 in base frame
    #   Blue team (base at Y = +0.990 in world): turntable at Y = -0.990 in base frame
    if team == 'red':
        TURNTABLE_CENTER = np.array([0.0, 0.990])  # In base frame
    else:
        TURNTABLE_CENTER = np.array([0.0, -0.990])  # In base frame
    TURNTABLE_RADIUS = 0.305  # meters
    TURNTABLE_HEIGHT = 0.200  # meters
    BLOCK_HALF_HEIGHT = 0.025  # 50mm cube / 2

    # Move to observation position
    print("Moving to observation position over turntable...")
    arm.safe_move_to_position(observation_position)
    q_in = observation_position
    print("Reached observation position.")

    # Get camera transform at observation pose
    H_ee_camera = detector.get_H_ee_camera()
    _, T_base_ee = fk_solver.forward(q_in)


    # Keep attempting dynamic blocks until time runs out or we grab enough
    MAX_DYNAMIC_GRABS = 4  # Maximum dynamic blocks to grab
    dynamic_grabbed = 0
    attempt = 0

    while dynamic_grabbed < MAX_DYNAMIC_GRABS:
        attempt += 1
        print(f"\n--- Dynamic Attempt {attempt} ---")

        # Scan for any dynamic block
        _, T_base_ee = fk_solver.forward(q_in)
        detections = detector.get_detections()
        t_now = time_in_seconds()

        target_samples = []
        found_block = False

        for (name, pose) in detections:
            if "dynamic" in name:
                H_world_block = T_base_ee @ H_ee_camera @ pose
                x = H_world_block[0, 3]
                y = H_world_block[1, 3]
                phi = np.arctan2(y - TURNTABLE_CENTER[1], x - TURNTABLE_CENTER[0])
                theta_block = np.arctan2(H_world_block[1, 0], H_world_block[0, 0])
                target_samples.append((t_now, x, y, theta_block, name))
                print(f"  Found dynamic block: x={x:.3f}, y={y:.3f}, phi={np.degrees(phi):.1f}°")
                found_block = True
                break

        if not found_block:
            print("No dynamic block detected. Retrying...")
            continue

        # Step 2: Track the block to estimate angular velocity
        NUM_SAMPLES = 8

        for i in range(NUM_SAMPLES):
            _, T_base_ee = fk_solver.forward(q_in)
            detections = detector.get_detections()
            t_now = time_in_seconds()

            for (name, pose) in detections:
                if "dynamic" in name:
                    H_world_block = T_base_ee @ H_ee_camera @ pose
                    x = H_world_block[0, 3]
                    y = H_world_block[1, 3]
                    theta_block = np.arctan2(H_world_block[1, 0], H_world_block[0, 0])
                    target_samples.append((t_now, x, y, theta_block, name))
                    break

        if len(target_samples) < 3:
            print("Not enough samples to estimate motion. Retrying...")
            continue

        # Step 3: Estimate angular velocity from position changes
        phis = []
        times = []
        for (t, x, y, theta_b, name) in target_samples:
            phi = np.arctan2(y - TURNTABLE_CENTER[1], x - TURNTABLE_CENTER[0])
            phis.append(phi)
            times.append(t)

        # Unwrap angles to handle wraparound
        phis = np.unwrap(phis)
        times = np.array(times)

        # Linear fit to get angular velocity (omega = d(phi)/dt)
        if len(times) > 1:
            A = np.vstack([times, np.ones(len(times))]).T
            omega, phi0 = np.linalg.lstsq(A, phis, rcond=None)[0]
        else:
            print("Not enough data for velocity estimation.")
            continue

        # Get current block position and radius
        last_sample = target_samples[-1]
        t_last, x_last, y_last, theta_last, block_name = last_sample
        radius = np.sqrt((x_last - TURNTABLE_CENTER[0])**2 + (y_last - TURNTABLE_CENTER[1])**2)

        print(f"Tracking: omega={np.degrees(omega):.1f} deg/s, r={radius:.3f}m")

        # Step 4: Predict intercept position
        ARM_MOVE_TIME = 2.5  # seconds - adjust based on testing

        t_predict = time_in_seconds() + ARM_MOVE_TIME
        phi_predict = omega * t_predict + phi0

        # Calculate predicted position
        x_predict = TURNTABLE_CENTER[0] + radius * np.cos(phi_predict)
        y_predict = TURNTABLE_CENTER[1] + radius * np.sin(phi_predict)
        z_predict = TURNTABLE_HEIGHT + BLOCK_HALF_HEIGHT

        print(f"Predicted intercept: x={x_predict:.3f}, y={y_predict:.3f}, phi={np.degrees(phi_predict):.1f}°")

        # If predicted position is not on our side, calculate when it will be
        # and adjust ARM_MOVE_TIME accordingly
        if team == 'red' and y_predict > 0.05:
            # Need to wait for block to rotate to Y < 0 side
            # Find phi where y = 0 (phi = 0 or -pi)
            # For red team, we want phi ~ -pi/2 (y negative, approaching)
            target_phi = -pi/2
            time_to_target = (target_phi - (omega * t_predict + phi0 - omega * t_predict)) / omega
            if time_to_target < 0:
                time_to_target += 2 * pi / abs(omega)  # Wait for next rotation
            t_predict = time_in_seconds() + max(ARM_MOVE_TIME, time_to_target)
            phi_predict = omega * t_predict + phi0
            x_predict = TURNTABLE_CENTER[0] + radius * np.cos(phi_predict)
            y_predict = TURNTABLE_CENTER[1] + radius * np.sin(phi_predict)
            print(f"Adjusted intercept: x={x_predict:.3f}, y={y_predict:.3f}")
        elif team == 'blue' and y_predict < -0.05:
            # Need to wait for block to rotate to Y > 0 side
            target_phi = pi/2
            time_to_target = (target_phi - (omega * t_predict + phi0 - omega * t_predict)) / omega
            if time_to_target < 0:
                time_to_target += 2 * pi / abs(omega)
            t_predict = time_in_seconds() + max(ARM_MOVE_TIME, time_to_target)
            phi_predict = omega * t_predict + phi0
            x_predict = TURNTABLE_CENTER[0] + radius * np.cos(phi_predict)
            y_predict = TURNTABLE_CENTER[1] + radius * np.sin(phi_predict)
            print(f"Adjusted intercept: x={x_predict:.3f}, y={y_predict:.3f}")

        # Step 4: Compute grasp pose
        # Gripper should align with block's predicted orientation
        # Block orientation also rotates with turntable
        theta_predict = theta_last + omega * (t_predict - t_last)

        # Build grasp matrices (same logic as static blocks)
        c, s = np.cos(theta_predict), np.sin(theta_predict)
        R_z_yaw = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        R_x_180 = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        T_pre_grasp_dyn = np.identity(4)
        T_pre_grasp_dyn[:3, :3] = R_z_yaw @ R_x_180
        T_pre_grasp_dyn[:3, 3] = np.array([x_predict, y_predict, z_predict + 0.08])

        T_grasp_dyn = deepcopy(T_pre_grasp_dyn)
        T_grasp_dyn[2, 3] = z_predict

        # Step 5: Execute grasp
        print("Computing IK for dynamic grasp...")
        q_pre_dyn, _, success_p, _ = ik_solver.inverse(T_pre_grasp_dyn, q_in, 'J_pseudo', 0.5)

        if not success_p:
            print("  [IK WARN] Trying 180-degree symmetry...")
            R_symmetry = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            T_pre_grasp_dyn = T_pre_grasp_dyn @ R_symmetry
            T_grasp_dyn = T_grasp_dyn @ R_symmetry
            q_pre_dyn, _, success_p, _ = ik_solver.inverse(T_pre_grasp_dyn, q_in, 'J_pseudo', 0.5)

        if not success_p:
            print("Pre-grasp IK failed for dynamic block. Retrying...")
            continue

        # Open gripper and move to pre-grasp
        arm.exec_gripper_cmd(0.080)
        arm.safe_move_to_position(q_pre_dyn)
        q_in = q_pre_dyn

        # Quick re-detect and adjust before final grasp
        _, T_base_ee = fk_solver.forward(q_in)
        detections = detector.get_detections()
        final_pose = None
        for (name, pose) in detections:
            if "dynamic" in name:
                H_world_block = T_base_ee @ H_ee_camera @ pose
                final_pose = H_world_block
                break

        if final_pose is not None:
            # Update grasp position with fresh detection
            T_grasp_dyn[:3, 3] = final_pose[:3, 3]
            T_grasp_dyn[2, 3] = TURNTABLE_HEIGHT + BLOCK_HALF_HEIGHT

        # Grasp IK
        q_grasp_dyn, _, success_g, _ = ik_solver.inverse(T_grasp_dyn, q_in, 'J_pseudo', 0.5)

        if not success_g:
            R_symmetry = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
            T_grasp_dyn_sym = T_grasp_dyn @ R_symmetry
            q_grasp_dyn, _, success_g, _ = ik_solver.inverse(T_grasp_dyn_sym, q_in, 'J_pseudo', 0.5)

        if not success_g:
            print("Grasp IK failed for dynamic block.")
            continue

        # Execute grasp
        arm.safe_move_to_position(q_grasp_dyn)
        arm.exec_gripper_cmd(0.03, 50)

        # Check if we grabbed something
        gripper_state = arm.get_gripper_state()
        if gripper_state['position'][0] < 0.01:
            print("Gripper closed fully - likely missed the block.")
            arm.exec_gripper_cmd(0.080)
            continue

        print("Block grabbed! Moving to place position...")

        # Place the dynamic block (same target as static)
        if team == 'red':
            table_pos_base = np.array([0.562, 0.075, 0.200])
        else:
            table_pos_base = np.array([0.562, -0.075, 0.200])

        # Apply the same placement offsets
        table_pos_base[0] += PLACE_OFFSET_X
        table_pos_base[1] += PLACE_OFFSET_Y

        z_target = 0.200 + (num_stacked * 0.05) + 0.025

        T_pre_place = np.eye(4)
        T_pre_place[[1, 2], [1, 2]] = -1
        T_pre_place[:3, 3] = table_pos_base
        T_pre_place[2, 3] = z_target + 0.06

        T_place = deepcopy(T_pre_place)
        T_place[2, 3] = z_target + 0.005

        # Try multiple seeds for place IK
        place_seeds = [q_in, scan_position, start_position]
        success_rp = False
        for seed in place_seeds:
            q_pre_place, _, success_rp, _ = ik_solver.inverse(T_pre_place, seed, 'J_pseudo', 0.5)
            if success_rp:
                break

        if not success_rp:
            print("Pre-place IK failed!")

        if success_rp:
            arm.safe_move_to_position(q_pre_place)
            q_in = q_pre_place

            q_place, _, success_r, _ = ik_solver.inverse(T_place, q_in, 'J_pseudo', 0.5)
            if success_r:
                arm.safe_move_to_position(q_place)
                arm.exec_gripper_cmd(0.080)
                arm.safe_move_to_position(q_pre_place)
                q_in = q_pre_place

                print(f"Dynamic block placed! (Stack {num_stacked})")
                num_stacked += 1
                dynamic_grabbed += 1

        # Return to observation position for next attempt
        arm.safe_move_to_position(observation_position)
        q_in = observation_position

    print(f"\n=== FINAL: Stacked {num_stacked} total blocks ({dynamic_grabbed} dynamic) ===")

    # END STUDENT CODE
