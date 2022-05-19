import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from gurobipy import*
import math

def continuous_ECBF_version1(x, action, f, g):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5  # actually half the pole's length
    polemass_length = (masspole * length)
    torque_bound = 1000
    theta = x[0]
    theta_dot = x[1]
    position = x[2]
    position_dot = x[3]

    m = Model('ECBF')
    m.remove(m.getConstrs())
    soft = m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="soft_feasible")
    u_CBF = m.addVar(lb=-torque_bound, ub=torque_bound, vtype=GRB.CONTINUOUS, name="safe_controller")
    K = 1000
    cost_func =  (float(action)-u_CBF)**2 + K*soft
    m.setObjective(cost_func, GRB.MINIMIZE)

    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (u_CBF + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    H = -theta**2 + (12 * 2 * math.pi / 360)**2
    H_d = -2*theta*theta_dot
    H_dd = -2*theta_dot**2 - 2*theta*thetaacc
    k1 = 55.#1.32
    k2 = 15.5#2.3
    m.addConstr((H_dd + k1 * H + k2 * H_d + soft) >= 0, name='CBF angle Constraint')

    H1 = -position**2 + 2.4 ** 2
    H1_d = -2*position*position_dot
    H1_dd = -2*position_dot**2 - 2*position*xacc
    m.addConstr((H1_dd + k1 * H1 + k2 * H1_d + soft) >= 0, name='CBF position Constraint')

    m.Params.LogToConsole = 0
    m.optimize()
    X = u_CBF.X
    slack = soft.X
    return np.expand_dims(np.array(X), 0), np.expand_dims(np.array(slack), 0)

def discrete_ECBF(x, action, f, g):
    dt = 0.05
    G = 10.
    mass = 1.0
    l = 1.0

    torque_bound = 50
    max_speed = 30
    theta = float(x[0])
    theta_dot = float(x[1])

    m = Model('ECBF')
    m.remove(m.getConstrs())
    soft = m.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS, name="soft_feasible")
    u_CBF = m.addVar(lb=-torque_bound, ub=torque_bound, vtype=GRB.CONTINUOUS, name="safe_controller")
    K = 1000
    cost_func =  (float(action)-u_CBF)**2 + K*soft
    m.setObjective(cost_func, GRB.MINIMIZE)

    thetaacc = -3 * G / (2 * l) * math.sin(theta+math.pi) +  3. / (mass * 1 ** 2) * u_CBF

    H = -theta ** 2 + 1 ** 2
    H_d = -2 * theta * thetaacc
    k1 = 15.  # 1.32
    m.addConstr( (H_d + k1*H + soft)>= 0, name='CBF angle Constraint')

    newthdot = theta_dot + thetaacc*dt
    H_s = -newthdot ** 2 + max_speed ** 2
    m.addConstr( H_s >= 0, name='Max speed Constraint')

    m.Params.LogToConsole = 0
    m.optimize()
    X = u_CBF.X
    slack = soft.X
    return np.expand_dims(np.array(X), 0), np.expand_dims(np.array(slack), 0)

def ECBF_control(x, action, f, g):

    N = len(action)
    P = matrix(np.diag([1., 1e24]), tc='d')
    q = matrix(np.zeros(N+1))
    H1 = np.array([1, 0.05])
    H2 = np.array([1, -0.05])
    H3 = np.array([-1, 0.05])
    H4 = np.array([-1, -0.05])
    F = math.pi/2 #1
    torque_bound = 15
    max_speed = 60

    gamma_b = 0.5
    G = np.array(
        [[-np.dot(H1, g), -np.dot(H4, g), 1, -1, g[1], -g[1]],
         [-1, -1, 0, 0, 0, 0]])
    G = np.transpose(G)

    h = np.array([gamma_b*F + np.dot(H1,f) - (1-gamma_b)*np.dot(H1,x),
                  gamma_b*F + np.dot(H4,f) - (1-gamma_b)*np.dot(H4,x),
                  -action + torque_bound,
                  action + torque_bound,
                  -f[1] - g[1]*action + max_speed,
                  f[1] + g[1]*action + max_speed])
    h = np.squeeze(h).astype(np.double)

    G = matrix(G,tc='d')
    h = matrix(h,tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u_bar = sol['x']

    if (np.add(np.squeeze(action), np.squeeze(u_bar[0])) - 0.001 >= torque_bound):
        u_bar[0] = torque_bound - action
        print("Error in QP")
    elif (np.add(np.squeeze(action), np.squeeze(u_bar[0])) + 0.001 <= -torque_bound):
        u_bar[0] = -torque_bound - action
        print("Error in QP")
    else:
        pass

    return np.expand_dims(np.array(u_bar[0]), 0), np.expand_dims(np.array(u_bar[1]), 0)

def GP_ECBF_control(u_rl, f, g, x, std):

    N = len(u_rl)
    P = matrix(np.diag([1., 1e24]), tc='d')
    q = matrix(np.zeros(N+1))
    H1 = np.array([1, 0.05])
    H2 = np.array([1, -0.05])
    H3 = np.array([-1, 0.05])
    H4 = np.array([-1, -0.05])
    F = math.pi/2 #1
    torque_bound = 15
    max_speed = 60

    gamma_b = 0.5

    kd = 1.5
    u_a = 0

    G = np.array(
        [[-np.dot(H1, g), -np.dot(H2, g), -np.dot(H3, g), -np.dot(H4, g), 1, -1, g[1], -g[1]],
         [-1, -1, -1, -1, 0, 0, 0, 0]])
    G = np.transpose(G)

    h = np.array([gamma_b * F + np.dot(H1, f) + np.dot(H1, g) * u_a - (1 - gamma_b) * np.dot(H1, x) - kd * np.dot(np.abs(H1), std),
                  gamma_b * F + np.dot(H2, f) + np.dot(H2, g) * u_a - (1 - gamma_b) * np.dot(H2, x) - kd * np.dot(np.abs(H2), std),
                  gamma_b * F + np.dot(H3, f) + np.dot(H3, g) * u_a - (1 - gamma_b) * np.dot(H3, x) - kd * np.dot( np.abs(H3), std),
                  gamma_b * F + np.dot(H4, f) + np.dot(H4, g) * u_a - (1 - gamma_b) * np.dot(H4, x) - kd * np.dot(np.abs(H4), std),
                  -u_rl + torque_bound,
                  u_rl + torque_bound,
                  -f[1] - g[1] * u_rl + max_speed,
                  f[1] + g[1] * u_rl + max_speed])
    h = np.squeeze(h).astype(np.double)

    G = matrix(G, tc='d')
    h = matrix(h, tc='d')

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    u_bar = sol['x']

    if (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) - 0.001 >= torque_bound):
        u_bar[0] = torque_bound - u_rl
        print("Error in QP")
    elif (np.add(np.squeeze(u_rl), np.squeeze(u_bar[0])) + 0.001 <= -torque_bound):
        u_bar[0] = - torque_bound - u_rl
        print("Error in QP")
    else:
        pass

    return np.expand_dims(np.array(u_bar[0]), 0), np.expand_dims(np.array(u_bar[1]), 0)