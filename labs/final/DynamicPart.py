# This code can move the robot to the observation position and compute the rotation radius


import sys
import numpy as np
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments
from core.interfaces import ArmController, ObjectDetector

# For timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds


if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print("Team must be red or blue - make sure you are running final.launch!")
        sys.exit(1)

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    # Start configuration
    start_position = np.array([
        -0.01779206,
        -0.76012354,
         0.01978261,
        -2.34205014,
         0.02984053,
         1.54119353 + pi / 2.0,
         0.75344866,
    ])
    arm.safe_move_to_position(start_position) # on your mark!


    # Final observation poses already been measured, no meed IK to compute everytime (over the turntable, camera looking down)
    observation_position_red = np.array([
        -0.66902174, -0.80879801,  1.61876292,
        -1.13930242,  0.84472293,  1.31064934,  2.04106599
    ])

    observation_position_blue = np.array([
         0.67380074, -0.73944342, -1.71271262,
        -1.14979493, -0.75195756,  1.35414952, -0.49852892
    ])

    print("\n****************")
    if team == "blue":
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")

    # ======================
    #       STUDENT CODE
    # ======================

    # Move to observation position
    if team == "red":
        q_obs = observation_position_red
    else:
        q_obs = observation_position_blue

    print("Moving to observation pose over the turntable...")
    arm.safe_move_to_position(q_obs)
    print("Reached observation pose.")

    # Get camera-to-EE (for debugging)
    H_ee_camera = detector.get_H_ee_camera()
    print("H_ee_camera:\n", H_ee_camera)

    # Rotation center
    center_x = 0.0
    center_y = 0.0

    print("\nCollecting block positions to compute rotation radius...\n")

    NUM_SAMPLES = 20        # number of samples to gather
    SAMPLE_DT  = 0.05       # sampling interval (seconds)

    radii = []

    for i in range(NUM_SAMPLES):
        detections = detector.get_detections()

        if len(detections) == 0:
            print(f"[{i}] No detections.")
            rospy.sleep(SAMPLE_DT)
            continue

        # Try to find block
        target_pose = None
        for (name, pose) in detections:
            if "cube" in name:   # adjust if needed
                target_pose = pose
                break

        if target_pose is None:
            print(f"[{i}] No cube detected.")
            rospy.sleep(SAMPLE_DT)
            continue

        # Extract 3D position of the block
        p = target_pose[:3, 3]
        x, y = float(p[0]), float(p[1])

        # Compute radius relative to (0,0,0)
        dx = x - center_x
        dy = y - center_y
        r = np.sqrt(dx*dx + dy*dy)

        radii.append(r)

        print(f"[{i}] x = {x:.4f}, y = {y:.4f}, radius r = {r:.4f}")

        rospy.sleep(SAMPLE_DT)

    # 4) Print final estimated rotation radius
    print("\n==============================")
    if len(radii) > 0:
        r_mean = float(np.mean(radii))
        print(f"Estimated rotation radius (mean r): {r_mean:.4f} m")
    else:
        print("No valid radius readings collected.")
    print("==============================\n")

    # ======================
    #     END STUDENT CODE
    # ======================

