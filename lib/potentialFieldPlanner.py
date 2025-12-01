
import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy

from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from lib.rrt import rrt

class PotentialFieldPlanner:

    # Joint limits for Panda arm
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

    center = lower + (upper - lower) / 2  # center of range for each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Construct a potential field planner with solver parameters.

        Parameters:
        ----------
        tol : float
            Termination tolerance in joint space (distance between q and goal).
        max_steps : int
            Maximum number of iterations for the main planner loop.
        min_step_size : float
            Minimum step norm; below this we consider the gradient to have "vanished".
        """
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def attractive_force(target, current):
        """
        Compute the attractive force between current and target in workspace
        for one joint (3D point).

        Parameters:
        ----------
        target : (3,1) ndarray
            Desired position of a joint in workspace.
        current : (3,1) ndarray
            Current position of the same joint in workspace.

        Returns:
        -------
        att_f : (3,1) ndarray
            Attractive force vector in workspace.
        """
        d = 0.12
        att_f = np.zeros((3, 1))

        vec = target - current
        dist = np.linalg.norm(vec)

        # Quadratic near the target, conic further away
        if dist <= d:
            att_f = vec
        else:
            att_f = vec / dist

        return att_f

    def repulsive_force(self, obstacle, current, unitvec=np.zeros((3, 1))):
        """
        Compute the repulsive force exerted by one obstacle on one joint.

        Parameters:
        ----------
        obstacle : (6,) ndarray
            Box definition [xmin, ymin, zmin, xmax, ymax, zmax].
        current : (3,1) ndarray
            Current joint position in workspace.

        Returns:
        -------
        rep_f : (3,1) ndarray
            Repulsive force vector in workspace.
        """
        eta = 0.01
        rho_0 = 0.3

        rep_f = np.zeros((3, 1))

        # Distance and direction from point to box
        rho, unitvec = self.dist_point2box(current.T, obstacle.flatten())
        rho = float(rho[0])
        unitvec = unitvec.flatten()

        if rho < 1e-9:
            return np.zeros((3, 1))

        if 0 < rho <= rho_0:
            term1 = (1.0 / rho) - (1.0 / rho_0)
            term2 = 1.0 / (rho ** 2)
            rep_f = eta * term1 * term2 * (-1.0 * unitvec)
            rep_f = rep_f.reshape((3, 1))
        else:
            rep_f = np.zeros((3, 1))

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Compute the distance and direction from a point to an axis-aligned box.

        Parameters:
        ----------
        p : (N,3) ndarray
            Points in workspace.
        box : (6,) ndarray
            Box definition [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns:
        -------
        dist : (N,) ndarray
            Distances from each point to the box.
        unit : (N,3) ndarray
            Unit vector pointing from point to the closest point on the box.
        """
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = 0.5 * (boxMin + boxMax)
        p = np.array(p)

        dx = np.amax(
            np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T,
            axis=1,
        )
        dy = np.amax(
            np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T,
            axis=1,
        )
        dz = np.amax(
            np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T,
            axis=1,
        )

        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        signs = np.sign(boxCenter - p)
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    def compute_forces(self, target, obstacle, current):
        """
        Compute the total workspace forces (attractive + repulsive) on every joint.

        Parameters:
        ----------
        target : (3,9) ndarray
            Desired positions (workspace) of 9 points along the robot.
        obstacle : list or ndarray
            List of obstacles, each given as a 6D box.
        current : (3,9) ndarray
            Current positions (workspace) of the same 9 points.

        Returns:
        -------
        joint_forces : (3,9) ndarray
            Total workspace forces on each of the 9 points.
        """
        joint_forces = np.zeros((3, 9))

        zeta_physical = 30.0
        zeta_virtual = 15.0

        n_obstacles = len(obstacle) if obstacle is not None else 0

        for i in range(9):
            target_i = target[:, i:i+1]
            current_i = current[:, i:i+1]

            # Choose gain for physical vs virtual links
            if i <= 6:
                zeta = zeta_physical
            else:
                zeta = zeta_virtual

            F_att_i = zeta * self.attractive_force(target_i, current_i)

            F_rep_total = np.zeros((3, 1))
            for j in range(n_obstacles):
                obs = obstacle[j]
                obs_arr = np.array(obs).flatten()
                F_rep = self.repulsive_force(obs_arr, current_i)
                F_rep_total += F_rep

            F_total_i = F_att_i + F_rep_total
            joint_forces[:, i] = F_total_i.flatten()

        return joint_forces

    def compute_torques(self, joint_forces, q):
        """
        Convert workspace forces on each point into joint torques.

        Parameters:
        ----------
        joint_forces : (3,9) ndarray
            Forces acting on each of the 9 link points.
        q : (7,) or (1,7) ndarray
            Current joint configuration.

        Returns:
        -------
        total_torque : (1,9) ndarray
            Torques for 9 joints (we will only use first 7 entries).
        """
        # total_torque = np.zeros((7, 1))

        # for i in range(9):
        #     F_i = joint_forces[:, i:i+1]
        #     Jvi = self.fk.Jv_i(q, i)     # (3,7)
        #     Jvi = np.asarray(Jvi)[:, :7]
        #     tau_i = Jvi.T @ F_i          # (7,1)
        #     total_torque += tau_i

        # return total_torque.T

        total_dq = np.zeros((7, 1))
        q_flat = q.flatten()

        for i in range(9):
            F_i = joint_forces[:, i:i+1] # (3,1)
            
            if np.linalg.norm(F_i) < 1e-6:
                continue

            Jvi = self.fk.Jv_i(q, i)[:, :7] # (3,7)
            
            # dq = pinv(J) * v_workspace
            try:
                J_pinv = np.linalg.pinv(Jvi) # (7,3)
                dq_i = J_pinv @ F_i          # (7,1)
                total_dq += dq_i
            except np.linalg.LinAlgError:
                pass

        return total_dq.T 

    @staticmethod
    def q_distance(target, current):
        """
        Euclidean distance between two joint configurations.

        Parameters:
        ----------
        target : (7,) or (1,7) ndarray
        current : (7,) or (1,7) ndarray

        Returns:
        -------
        distance : float
            L2 norm of difference in joint space.
        """
        diff = target - current
        distance = np.linalg.norm(diff)
        return distance

    ###########################
    ### Gradient Computation ###
    ###########################

    def compute_gradient(self, q, target, map_struct):
        """
        Compute the combined gradient in joint space:

        - A small, normalized joint-space attractive term (towards target).
        - A workspace-based potential field term converted into joint velocities.
        """
        q = np.asarray(q).reshape(1, 7)
        target = np.asarray(target).reshape(1, 7)

        # 1. Joint-space attraction (small, normalized step)
        joint_grad = target - q
        norm_j = np.linalg.norm(joint_grad)
        joint_step = 0.03  # max magnitude of the joint-space pull

        if norm_j > 1e-8:
            joint_grad = joint_grad * (joint_step / norm_j)
        else:
            joint_grad = np.zeros((1, 7))

        # If already very close to the target, rely only on joint-space pull
        if self.q_distance(target.flatten(), q.flatten()) < 0.05:
            return joint_grad

        # 2. Workspace potential field contribution 
        
        current_jps, _ = self.fk.forward_expanded(q.flatten())
        target_jps, _ = self.fk.forward_expanded(target.flatten())

        current_positions = current_jps[:9, :].T  # (3, 9)
        target_positions = target_jps[:9, :].T    # (3, 9)

        obstacles = map_struct.obstacles

        joint_forces = self.compute_forces(target_positions, obstacles, current_positions)

        dq_ws = self.compute_torques(joint_forces, q)

        # Normalize workspace contribution 

        ws_step = 0.1 
        norm_ws = np.linalg.norm(dq_ws)
        
        if norm_ws > ws_step and norm_ws > 0:
            dq_ws = dq_ws * (ws_step / norm_ws)

        # 3. Fade repulsion when close to the target 
        err = self.q_distance(q.flatten(), target.flatten())
        FADE_START = 0.6 
        FADE_END   = 0.1  

        if err >= FADE_START:
            w_ws = 1.0
        elif err <= FADE_END:
            w_ws = 0.1
        else:
            t = (err - FADE_END) / (FADE_START - FADE_END)
            w_ws = 0.1 + 0.9 * t

        # 4. Combine 
        dq = joint_grad + w_ws * dq_ws

        return dq

    def plan(self, map_struct, start, goal):
        """
        Global RRT (once) + Sparse milestones + Local APF tracking.

        1) Call RRT once in joint space to get a coarse global path.
        2) Down-sample that path to a few "milestones".
        3) Use APF (compute_gradient) to move from one milestone to the next.
        4) If RRT fails, fall back to pure APF from start to goal.

        Notes:
        - RRT code is NOT modified, we only call rrt(...) with a smaller max_iter.
        - No simulated annealing, no complicated escape logic, to keep runtime low.
        """

        q_current = np.asarray(start).reshape(1, 7)
        goal = np.asarray(goal).reshape(1, 7)

        goal_tol = self.tol
        max_iters = self.max_steps

        q_path = [q_current.copy()]
        obstacles = map_struct.obstacles

        # ---------------------------------------------------------
        # 0) Global RRT once, to get a coarse path in joint space
        # ---------------------------------------------------------
        try:
            rrt_path = rrt(map_struct,
                           q_current.flatten(),
                           goal.flatten(),
                           max_iter=5000)
        except Exception as e:
            # print("[RRT] Exception:", e)
            rrt_path = None

        milestones = []

        if rrt_path is not None and len(rrt_path) >= 2:
            rrt_path = np.asarray(rrt_path)

            # --- Down-sample to a few milestones ---
            n = rrt_path.shape[0]
            step_size = 5

            # if n <= 2:
            #     # No useful intermediate points
            #     milestones = [goal.copy()]
            # else:
            #     k = min(max_milestones, n - 2)
            #     # Evenly spaced indices between 1 and n-2 (inclusive)
            #     idxs = np.linspace(1, n - 2, k)
            #     idxs = np.round(idxs).astype(int)

            #     for idx in idxs:
            #         milestones.append(rrt_path[idx].reshape(1, 7))
            if n < step_size:
                idxs = np.arange(1, n-1)
            else:
                idxs = np.arange(step_size, n-1, step_size) # 简单的切片
            
            for idx in idxs:
                milestones.append(rrt_path[idx].reshape(1, 7))

                # Final target is always the true goal
                milestones.append(goal.copy())
        else:
            # RRT failed: just have a single target = goal
            # print("[RRT] Global RRT failed or path too short, falling back to pure APF")
            milestones = [goal.copy()]

        # ---------------------------------------------------------
        # 1) APF tracking of milestones
        # ---------------------------------------------------------
        target_idx = 0
        current_target = milestones[target_idx]  # 1x7

        # Progress detector relative to the CURRENT target (not the final goal)
        current_error = self.q_distance(q_current.flatten(),
                                        current_target.flatten())
        min_error_so_far = current_error
        PROGRESS_THRESHOLD = 1e-3
        steps_since_progress = 0
        STUCK_THRESHOLD = 20

        for iteration in range(max_iters):

            # --- Distance to current target milestone ---
            current_error = self.q_distance(q_current.flatten(),
                                            current_target.flatten())

            # 1) Check whether we reached the current milestone
            #    For intermediate milestones: a slightly looser tolerance.
            if target_idx < len(milestones) - 1:
                # intermediate milestone
                subgoal_tol = max(goal_tol, 0.05)
            else:
                # final goal
                subgoal_tol = goal_tol

            if current_error < subgoal_tol:
                # Reached this milestone
                if target_idx == len(milestones) - 1:
                    # Last one: we are done
                    print(f"Goal Reached at step {iteration}! Error: {current_error}")
                    break
                else:
                    # Switch to next milestone
                    target_idx += 1
                    current_target = milestones[target_idx]
                    # Reset progress tracking for new target
                    min_error_so_far = self.q_distance(q_current.flatten(),
                                                       current_target.flatten())
                    steps_since_progress = 0
                    # Proceed to next iteration with the new target
                    continue

            # 2) Normal APF gradient toward the current milestone
            dq = self.compute_gradient(q_current, current_target, map_struct)

            # 2.1) Meaningful progress detection
            if current_error < (min_error_so_far - PROGRESS_THRESHOLD):
                min_error_so_far = current_error
                steps_since_progress = 0
            else:
                steps_since_progress += 1

            # 3) Simple "stuck" handling: random kick if no progress for a while
            if steps_since_progress > STUCK_THRESHOLD:
                # Random kick with moderate amplitude
                print("Stuck detected! Invoking Local RRT...")
    
                # try to use rrt to plan for this sub object
                try:
                    rescue_path = rrt(map_struct, q_current.flatten(), current_target.flatten(), max_iter=1000)
                except:
                    rescue_path = None
        
                if rescue_path is not None and len(rescue_path) > 1:
                    for q_rrt in rescue_path[1:]:
                        q_path.append(q_rrt.reshape(1,7))
                    q_current = rescue_path[-1].reshape(1,7)
                    steps_since_progress = 0
                    min_error_so_far = self.q_distance(q_current, current_target)
                    print("Local RRT rescued!")
                    continue # skip this apf calculation
                else:
                    dq = 0.5 * (np.random.rand(1, 7) - 0.5) 
                    steps_since_progress = 0
                min_error_so_far = current_error

            # 4) Step normalization (limit joint motion per iteration)
            if current_error > 1.0:
                max_step = 0.2
            elif current_error > 0.5:
                max_step = 0.1
            else:
                max_step = 0.05

            norm_dq = np.linalg.norm(dq)
            if norm_dq > max_step and norm_dq > 0:
                dq = dq * (max_step / norm_dq)

            # 5) Backtracking line search + collision check
            step_size = 1.0
            moved = False

            while step_size > 0.01:
                q_next = q_current + (dq * step_size)

                # 5.1 Joint limits
                if np.any(q_next < self.lower) or np.any(q_next > self.upper):
                    step_size *= 0.5
                    continue

                # 5.2 Collision check (using your existing forward_expanded + detectCollision)
                jointPositions, _ = self.fk.forward_expanded(q_next.flatten())
                pts = jointPositions[:9, :]

                collided = False
                for box in obstacles:
                    if any(detectCollision(pts[:-1, :], pts[1:, :], box)):
                        collided = True
                        break

                if collided:
                    step_size *= 0.5
                else:
                    q_current = q_next.copy()
                    q_path.append(q_current.copy())
                    moved = True
                    break

            # If moved == False, this iteration does nothing.
            # Next iteration might trigger "stuck" and random kick.

        final_error = self.q_distance(q_current.flatten(), goal.flatten())
        if final_error > goal_tol:
            try:
                local_rrt = rrt(
                    map_struct,
                    q_current.flatten(),
                    goal.flatten(),
                    max_iter=1500
                )
            except Exception:
                local_rrt = None

            if local_rrt is not None and len(local_rrt) >= 2:
                local_rrt = np.asarray(local_rrt)
                for q in local_rrt[1:]:
                    q = np.asarray(q).reshape(1, 7)

                    if np.any(q < self.lower) or np.any(q > self.upper):
                        break

                    jointPositions, _ = self.fk.forward_expanded(q.flatten())
                    pts = jointPositions[:9, :]
                    collided = False
                    for box in obstacles:
                        if any(detectCollision(pts[:-1, :], pts[1:, :], box)):
                            collided = True
                            break

                    if collided:
                        break

                    q_current = q.copy()
                    q_path.append(q_current.copy())

        return np.vstack(q_path)








################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    planner = PotentialFieldPlanner()

    # Example testing
    map_struct = loadmap("../maps/map4.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])

    # goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    # goal = np.array([0.2, 0.4, -0.6, -1.3, 0.8, 1.2, 0.1])

    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print(
            "iteration:", i,
            " q =", q_path[i, :],
            " error={error}".format(error=error)
        )

    print("q path: ", q_path)