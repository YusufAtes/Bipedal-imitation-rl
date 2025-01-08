import numpy as np

def biped_plant(dt, x, dx, torques, ramp_angle, y_g, h, f):
    M = 48
    m_1 = 7
    m_2 = 4
    l_1 = 0.5
    l_2 = 0.6
    g = 9.8
    b_1 = 10
    b_2 = 10
    b_k = 1000
    b_g = 1000
    k_k = 10000
    k_g = 10000
    l_1_2 = l_1 / 2
    l_2_2 = l_2 / 2

    I_1 = m_1 * l_1**2 / 12
    I_2 = m_2 * l_2**2 / 12
    MInv = 1 / M
    m_1Inv = 1 / m_1
    m_2Inv = 1 / m_2
    a = [1.5, 1, 1.5, 1.5, 3, 1.5, 3, 1.5]

    sin_x = np.sin(x)
    cos_x = np.cos(x)

    l_1_2_sin_x5 = l_1_2 * sin_x[5]
    l_1_2_cos_x5 = l_1_2 * cos_x[5]
    l_1_2_sin_x8 = l_1_2 * sin_x[8]
    l_1_2_cos_x8 = l_1_2 * cos_x[8]
    l_2_2_sin_x11 = l_2_2 * sin_x[11]
    l_2_2_cos_x11 = l_2_2 * cos_x[11]
    l_2_2_sin_x14 = l_2_2 * sin_x[14]
    l_2_2_cos_x14 = l_2_2 * cos_x[14]

    l_1_2_sin_x5_I_1 = -l_1_2_sin_x5 / I_1
    l_1_2_cos_x5_I_1 = -l_1_2_cos_x5 / I_1
    l_1_2_sin_x8_I_1 = -l_1_2_sin_x8 / I_1
    l_1_2_cos_x8_I_1 = -l_1_2_cos_x8 / I_1
    l_2_2_sin_x11_I_2 = -l_2_2_sin_x11 / I_2
    l_2_2_cos_x11_I_2 = -l_2_2_cos_x11 / I_2
    l_2_2_sin_x14_I_2 = -l_2_2_sin_x14 / I_2
    l_2_2_cos_x14_I_2 = -l_2_2_cos_x14 / I_2

    P_x = np.array([
        [MInv, 0, MInv, 0, 0, 0, 0, 0],
        [0, MInv, 0, MInv, 0, 0, 0, 0],
        [-m_1Inv, 0, 0, 0, m_1Inv, 0, 0, 0],
        [0, -m_1Inv, 0, 0, 0, m_1Inv, 0, 0],
        [l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0, l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0],
        [0, 0, -m_1Inv, 0, 0, 0, m_1Inv, 0],
        [0, 0, 0, -m_1Inv, 0, 0, 0, m_1Inv],
        [0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1, 0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1],
        [0, 0, 0, 0, -m_2Inv, 0, 0, 0],
        [0, 0, 0, 0, 0, -m_2Inv, 0, 0],
        [0, 0, 0, 0, l_2_2_sin_x11_I_2, l_2_2_cos_x11_I_2, 0, 0],
        [0, 0, 0, 0, 0, 0, -m_2Inv, 0],
        [0, 0, 0, 0, 0, 0, 0, -m_2Inv],
        [0, 0, 0, 0, 0, 0, l_2_2_sin_x14_I_2, l_2_2_cos_x14_I_2],
    ])

    C_x = np.array([
        [1, 0, -1, 0, -l_1_2_sin_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, -l_1_2_cos_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, -1, 0, -l_1_2_sin_x8, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, -1, -l_1_2_cos_x8, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, -l_1_2_sin_x5, 0, 0, 0, -1, 0, -l_2_2_sin_x11, 0, 0, 0],
        [0, 0, 0, 1, -l_1_2_cos_x5, 0, 0, 0, 0, -1, -l_2_2_cos_x11, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, -l_1_2_sin_x8, 0, 0, 0, -1, 0, -l_2_2_sin_x14],
        [0, 0, 0, 0, 0, 0, 1, -l_1_2_cos_x8, 0, 0, 0, 0, -1, -l_2_2_cos_x14],
    ])

    # Ramp calculations
    x_r = x[9] + l_2_2_cos_x11
    y_r = x[10] - l_2_2_sin_x11
    y_g_x_r = y_g(x_r, ramp_angle)
    if (y_r - y_g_x_r) < 0:
        y_d = y_g_x_r - y_r
        hh = y_d / np.sin(x[11])
        x_d = min(l_2, hh) * np.cos(x[11])
        F_g_1 = -k_g * (x_d) - b_g * (dx[9] - l_2_2_sin_x11 * dx[11])
        F_g_2 = -k_g * (y_r - y_g_x_r) + b_g * f(-(dx[10] - l_2_2_cos_x11 * dx[11]))
    else:
        F_g_1 = 0
        F_g_2 = 0

    # Left foot calculations
    x_l = x[12] + l_2_2_cos_x14
    y_l = x[13] - l_2_2_sin_x14
    y_g_x_l = y_g(x_l, ramp_angle)
    if (y_l - y_g_x_l) < 0:
        y_d = y_g_x_l - y_l
        hh = y_d / np.sin(x[14])
        x_d = min(l_2, hh) * np.cos(x[14])
        F_g_3 = -k_g * (x_d) - b_g * (dx[12] - l_2_2_sin_x14 * dx[14])
        F_g_4 = -k_g * (y_l - y_g_x_l) + b_g * f(-(dx[13] - l_2_2_cos_x14 * dx[14]))
    else:
        F_g_3 = 0
        F_g_4 = 0

    # Differential Equations
    f_x5_x11 = max(0, x[5] - x[11])
    f_x8_x14 = max(0, x[8] - x[14])
    
    h_x5_x11 = h(x[5] - x[11])
    h_x8_x14 = h(x[8] - x[14])

    T_r1_y = torques[0]
    T_r2_y = torques[1]
    T_r3_y = torques[2]
    T_r4_y = torques[3]
    T_r5_x_dx_y = torques[4]
    T_r6_x_dx_y = torques[5]

    Q_x_dx_y_F_g = np.array([
        0,
        -g,
        0,
        -g,
        (-b_1 * abs(x[5] - np.pi/2) * dx[5] - (b_2 + b_k * f_x5_x11) * (dx[5] - dx[11]) - k_k * h_x5_x11 + T_r1_y + T_r3_y) / I_1,
        0,
        -g,
        (-b_1 * abs(x[8] - np.pi/2) * dx[8] - (b_2 + b_k * f_x8_x14) * (dx[8] - dx[14]) - k_k * h_x8_x14 + T_r2_y + T_r4_y) / I_1,
        F_g_1 / m_2,
        F_g_2 / m_2 - g,
        (-F_g_1 * l_2_2_sin_x11 - F_g_2 * l_2_2_cos_x11 - (b_2 + b_k * f_x5_x11) * (dx[11] - dx[5]) + k_k * h_x5_x11 - T_r3_y - T_r5_x_dx_y) / I_2,
        F_g_3 / m_2,
        F_g_4 / m_2 - g,
        (-F_g_3 * l_2_2_sin_x14 - F_g_4 * l_2_2_cos_x14 - (b_2 + b_k * f_x8_x14) * (dx[14] - dx[8]) + k_k * h_x8_x14 - T_r4_y - T_r6_x_dx_y) / I_2
    ])

    dx5_2 = dx[5] ** 2
    dx8_2 = dx[8] ** 2
    dx11_2 = dx[11] ** 2
    dx14_2 = dx[14] ** 2

    l_1_2_cos_x5_dx5_2 = l_1_2_cos_x5 * dx5_2
    l_1_2_sin_x5_dx5_2 = l_1_2_sin_x5 * dx5_2
    l_1_2_cos_x8_dx8_2 = l_1_2_cos_x8 * dx8_2
    l_1_2_sin_x8_dx8_2 = l_1_2_sin_x8 * dx8_2

    D_x_dx = np.array([
        l_1_2_cos_x5_dx5_2,
        -l_1_2_sin_x5_dx5_2,
        l_1_2_cos_x8_dx8_2,
        -l_1_2_sin_x8_dx8_2,
        l_1_2_cos_x5_dx5_2 + l_2_2_cos_x11 * dx11_2,
        -l_1_2_sin_x5_dx5_2 - l_2_2_sin_x11 * dx11_2,
        l_1_2_cos_x8_dx8_2 + l_2_2_cos_x14 * dx14_2,
        -l_1_2_sin_x8_dx8_2 - l_2_2_sin_x14 * dx14_2
    ])

    # Calculate d2x
    d2x = P_x @ np.linalg.solve(C_x @ P_x, (D_x_dx - C_x @ Q_x_dx_y_F_g)) + Q_x_dx_y_F_g
    dx = dx + dt * d2x
    x = x + dt * dx

    # Feedback Calculations
    F_eed = np.zeros(12)
    h_F_g_2 = h(F_g_2)
    h_F_g_4 = h(F_g_4)
    F_eed[0] = a[0] * (x[5] - np.pi/2) - a[1] * (x[8] - np.pi/2) + a[2] * (x[11] - np.pi/2) * h_F_g_2 + a[3] * h_F_g_4
    F_eed[1] = -F_eed[0]
    F_eed[2] = a[0] * (x[8] - np.pi/2) - a[1] * (x[5] - np.pi/2) + a[2] * (x[14] - np.pi/2) * h_F_g_4 + a[3] * h_F_g_2
    F_eed[3] = -F_eed[2]
    F_eed[4] = a[4] * (np.pi/2 - x[14]) * h_F_g_4
    F_eed[5] = -F_eed[4]
    F_eed[6] = a[4] * (np.pi/2 - x[11]) * h_F_g_2
    F_eed[7] = -F_eed[6]
    F_eed[8] = a[5] * (np.pi/2 - x[11]) * h_F_g_2 + a[6] * (np.pi/2 - x[14]) * h_F_g_4 - a[7] * dx[11] * h_F_g_2
    F_eed[9] = -F_eed[8]
    F_eed[10] = a[5] * (np.pi/2 - x[14]) * h_F_g_4 + a[6] * (np.pi/2 - x[11]) * h_F_g_2 - a[7] * dx[14] * h_F_g_4
    F_eed[11] = -F_eed[10]

    return x, dx, F_eed
