import math
import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np

class BicycleEnv(gym.Env):
    """Bicycle balancing/riding domain from https://github.com/amarack/python-rl.
    From the paper:
    Learning to Drive a Bicycle using Reinforcement Learning and Shaping.
    Jette Randlov and Preben Alstrom. 1998.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):
        self.noise = kwargs.setdefault('noise', 0.04)
        self.random_start = kwargs.setdefault('random_start', False)

        self.state = np.zeros((5,)) # omega, omega_dot, omega_ddot, theta, theta_dot
        self.position = np.zeros((5,)) # x_f, y_f, x_b, y_b, psi

        self.state_range = np.array([[-np.pi * 12./180., np.pi * 12./180.],
                                        [-np.pi * 2./180., np.pi * 2./180.],
                                        [-np.pi, np.pi],
                                        [-np.pi * 80./180., np.pi * 80./180.],
                                        [-np.pi * 2./180., np.pi * 2./180.]])

        high = np.array([np.pi * 12./180.,
                         np.pi * 2./180.,
                         np.pi,
                         np.pi * 80./180.,
                         np.pi * 2./180.],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.max_torque = 2 
        self.max_displacement = 0.02
        high = np.array([self.max_torque, self.max_displacement])
        self.action_space = spaces.Box(low=-high, high=high, shape=(2,), dtype=np.float32)

        self.psi_range = np.array([-np.pi, np.pi])

        self.reward_fall = -1.0
        self.reward_goal = 0.01
        self.goal_rsqrd = 100.0 # Square of the radius around the goal (10m)^2
        self.navigate = kwargs.setdefault('navigate', False)
        if not self.navigate:
            # Original balancing task
            self.reward_shaping = 0.001
        else:
            self.reward_shaping = 0.00004

        self.goal_loc = np.array([1000., 0])

        # Units in Meters and Kilograms
        self.c = 0.66       # Horizontal dist between bottom of front wheel and center of mass
        self.d_cm = 0.30    # Vertical dist between center of mass and the cyclist
        self.h = 0.94       # Height of the center of mass over the ground
        self.l = 1.11       # Dist between front tire and back tire at point on ground
        self.M_c = 15.0     # Mass of bicycle
        self.M_d = 1.7      # Mass of tire
        self.M_p = 60       # Mass of cyclist
        self.r = 0.34       # Radius of tire
        self.v = 10.0 / 3.6 # Velocity of bicycle (converted from km/h to m/s)

        # Useful precomputations
        self.M = self.M_p + self.M_c
        self.Inertia_bc = (13./3.) * self.M_c * self.h**2 + self.M_p * (self.h + self.d_cm)**2
        self.Inertia_dv = self.M_d * self.r**2
        self.Inertia_dl = .5 * self.M_d * self.r**2
        self.sigma_dot = self.v / self.r

        # Simulation Constants
        self.gravity = 9.8
        self.delta_time = 0.02
        self.sim_steps = 10



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state.fill(0.0)
        self.position.fill(0.0)
        self.position[3] = self.l
        self.position[4] = np.arctan((self.position[1]-self.position[0])/(self.position[2] - self.position[3]))
        return self.state


    def step(self, action):
        T = np.clip(action[0], -self.max_torque, self.max_torque) # Torque on handle bars
        d = np.clip(action[1], -self.max_displacement, self.max_displacement) # Displacement of center of mass (in meters)
        if self.noise > 0:
            d += (np.random.random()-0.5)*self.noise # Noise between [-0.02, 0.02] meters

        omega, omega_dot, omega_ddot, theta, theta_dot = tuple(self.state)
        x_f, y_f, x_b, y_b, psi = tuple(self.position)

        for _ in range(self.sim_steps):
            if theta == 0: # Infinite radius tends to not be handled well
                r_f = r_b = r_CM = 1.e8
            else:
                r_f = self.l / np.abs(np.sin(theta))
                r_b = self.l / np.abs(np.tan(theta))
                r_CM = np.sqrt((self.l - self.c)**2 + (self.l**2 / np.tan(theta)**2))

            varphi = omega + np.arctan(d / self.h)

            omega_ddot = self.h * self.M * self.gravity * np.sin(varphi)
            omega_ddot -= np.cos(varphi) * (self.Inertia_dv * self.sigma_dot * theta_dot + np.sign(theta)*self.v**2*(self.M_d * self.r *(1./r_f + 1./r_b) + self.M*self.h/r_CM))
            omega_ddot /= self.Inertia_bc

            theta_ddot = (T - self.Inertia_dv * self.sigma_dot * omega_dot) / self.Inertia_dl

            df = (self.delta_time / float(self.sim_steps))
            omega_dot += df * omega_ddot
            omega += df * omega_dot
            theta_dot += df * theta_ddot
            theta += df * theta_dot

            # Handle bar limits (80 deg.)
            theta = np.clip(theta, self.state_range[3,0], self.state_range[3,1])

            # Update position (x,y) of tires
            front_term = psi + theta + np.sign(psi + theta)*np.arcsin(self.v * df / (2.*r_f))
            back_term = psi + np.sign(psi)*np.arcsin(self.v * df / (2.*r_b))
            x_f += -np.sin(front_term)
            y_f += np.cos(front_term)
            x_b += -np.sin(back_term)
            y_b += np.cos(back_term)

            # Handle Roundoff errors, to keep the length of the bicycle constant
            dist = np.sqrt((x_f-x_b)**2 + (y_f-y_b)**2)
            if np.abs(dist - self.l) > 0.01:
                x_b += (x_b - x_f) * (self.l - dist)/dist
                y_b += (y_b - y_f) * (self.l - dist)/dist

            # Update psi
            if x_f==x_b and y_f-y_b < 0:
                psi = np.pi
            elif y_f - y_b > 0:
                psi = np.arctan((x_b - x_f)/(y_f - y_b))
            else:
                psi = np.sign(x_b - x_f)*(np.pi/2.) - np.arctan((y_f - y_b)/(x_b-x_f))

        self.state = np.array([omega, omega_dot, omega_ddot, theta, theta_dot])
        self.position = np.array([x_f, y_f, x_b, y_b, psi])

        if np.abs(omega) > self.state_range[0,1]: # Bicycle fell over
            return self.state, -1.0, True, {}
        elif self.isAtGoal():
            return self.state, self.reward_goal, True, {}
        elif not self.navigate:
            return self.state, self.reward_shaping, False, {}
        else:
            goal_angle = vector_angle(self.goal_loc, np.array([x_f-x_b, y_f-y_b])) * np.pi / 180.
            return self.state, (4. - goal_angle**2) * self.reward_shaping, False, {}

    def isAtGoal(self):
        # Anywhere in the goal radius
        if self.navigate:
            return np.sqrt(max(0.,((self.position[:2] - self.goal_loc)**2).sum() - self.goal_rsqrd)) < 1.e-5
        else:
            return False

    def render(self, mode='human', close=False):
        pass



def vector_angle(u, v):
    return np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)))*180.0/np.pi