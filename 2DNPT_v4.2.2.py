"7 overlapping disks consisting of a rod-like polymer, using NPT. Version 4.2.2"

import argparse

parser = argparse.ArgumentParser(description='2d polymer nematic order')
parser.add_argument('-s', '--P', nargs='?', default= 10 ** 5, help='2d polymer changing pressure', type=float)

args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
#import imageio
from random import randint

def calc_energy(n, x, y, theta, xx, yy, tt, T):
    global sigma, L
    e = 10 * T
    sigma = 1.0
    E = 0
    xs = []
    ys = []
    xxs = []
    yys = []
    for i in range(7):
        xxs.append(xx + (i - 3) * 0.8 * sigma * np.cos(tt))
        yys.append(yy + (i - 3) * 0.8 * sigma * np.sin(tt))
        if xxs[i] > L:
            xxs[i] -= L
        if xxs[i] < 0:
            xxs[i] += L
        if yys[i] > L:
            yys[i] -= L
        if yys[i] < 0:
            yys[i] += L
    for i in range(n):
        xss = []
        yss = []
        for j in range(7):
            xss.append(x[i] + (j - 3) * 0.8 * sigma * np.cos(theta[i]))
            yss.append(y[i] + (j - 3) * 0.8 * sigma * np.sin(theta[i]))
            if xss[j] > L:
                xss[j] -= L
            if xss[j] < 0:
                xss[j] += L
            if yss[j] > L:
                yss[j] -= L
            if yss[j] < 0:
                yss[j] += L
        xs.append(xss)
        ys.append(yss)

    rx = []
    ry = []
    for i in range(n):
        for j in range(7):
            for k in range(7):
                rx.append(min(abs(xxs[j] - xs[i][k]), L - abs(xxs[j] - xs[i][k])))
                ry.append(min(abs(yys[j] - ys[i][k]), L - abs(yys[j] - ys[i][k])))

    r = []
    for i in range(49 * n):
        r.append(float(np.sqrt(rx[i] ** 2 + ry[i] ** 2)))

    for i in range(49 * n):
        if r[i] <= 2 ** (1 / 6) * sigma:
            E += float(4 * e * ((2 ** (1 / 6) * sigma / r[i]) ** 12 - (2 ** (1 / 6) * sigma / r[i]) ** 6))
        if r[i] > 2 ** (1 / 6) * sigma:
            E += 0.0
    return E


def mc_init(L, NP, T):
    global kb
    kb = 1
    x = []
    y = []
    theta = []
    x0 = np.random.random() * L                             # random dot in a square as a center of mass of one molecule
    y0 = np.random.random() * L
    theta0 = np.random.random() * np.pi
    x.append(x0)
    y.append(y0)
    theta.append(theta0)
    totalE = 0
    i = 0
    while i < NP - 1:                                       # adding new molecules into the system
        xx = np.random.random() * L                         # NP molecules in total
        yy = np.random.random() * L
        tt = np.random.random() * np.pi
        E = calc_energy(i + 1, x, y, theta, xx, yy, tt, T)
        if E <= 0:
            x.append(xx)
            y.append(yy)
            theta.append(tt)
        elif E > 0:
            p = np.random.random()
            if p < np.exp(- E / (kb * T)):
                x.append(xx)
                y.append(yy)
                theta.append(tt)
                totalE += E
            elif p >= np.exp(- E / (kb * T)):
                i -= 1
        i += 1
    return x, y, theta


def mc_update(dl, dr, dtheta, x, y, theta, NP, T, P):
    global nei, successr, attemptr, successt, attemptt, succ, atte, L, total_e, total_energy, delta_e, total_e01
    i = 0
    xt = np.zeros(NP)
    yt = np.zeros(NP)
    thetat = np.zeros(NP)
    for j in range(NP):                                 #### v4.2.2 update ####
        xt[j] = x[j]
        yt[j] = y[j]
        thetat[j] = theta[j]
    delta_e = 0.0
    while i < NP:                                                   # random deviation of molecules
        dx = np.random.normal(scale=0.333) * abs(dr[i])
        dy = np.random.normal(scale=0.333) * abs(dr[i])
        dt = np.random.normal(scale=0.333) * abs(dtheta[i])
        xx = x[i] + dx
        yy = y[i] + dy
        tt = theta[i] + dt
        xx0 = x[i]
        yy0 = y[i]
        tt0 = theta[i]
        if xx > L:
            xx -= L
        if xx < 0:
            xx += L
        if yy > L:
            yy -= L
        if yy < 0:
            yy += L
        xinner = []
        yinner = []
        tinner = []
        xinner0 = []
        yinner0 = []
        tinner0 = []

        total_e01 = 0.0                           # minor bug fix
        for k in range(NP):                       #### v4.2.1 update #### calculate energy before each molecule's move
            xin0 = []
            yin0 = []
            tin0 = []
            if len(nei[k]) != 0:  # calculate potential energy from its neighbour molecules
                for j in range(len(nei[k])):
                    rx0 = min(abs(xt[nei[k][j]] - xt[k]), L - abs(xt[nei[k][j]] - xt[k]))
                    ry0 = min(abs(yt[nei[k][j]] - yt[k]), L - abs(yt[nei[k][j]] - yt[k]))
                    if np.sqrt(rx0 ** 2 + ry0 ** 2) < 5.8 * sigma:
                        xin0.append(xt[nei[k][j]])
                        yin0.append(yt[nei[k][j]])
                        tin0.append(thetat[nei[k][j]])
                if len(xin0) != 0:
                    total_e01 += calc_energy(len(xin0), xin0, yin0, tin0, xt[k], yt[k], thetat[k], T) / 2.0
        print('TOTAL ENERGY 0 =', total_e01)

        if len(nei[i]) != 0:                                 # calculate potential energy from its neighbour molecules
            for j in range(len(nei[i])):
                rx0 = min(abs(xt[nei[i][j]] - xx0), L - abs(xt[nei[i][j]] - xx0))
                ry0 = min(abs(yt[nei[i][j]] - yy0), L - abs(yt[nei[i][j]] - yy0))
                if np.sqrt(rx0 ** 2 + ry0 ** 2) < 5.8 * sigma:
                    xinner0.append(xt[nei[i][j]])
                    yinner0.append(yt[nei[i][j]])
                    tinner0.append(thetat[nei[i][j]])
            if len(xinner0) != 0:
                Er0 = calc_energy(len(xinner0), xinner0, yinner0, tinner0, xx0, yy0, tt0, T)

            else:
                Er0 = 0

            for j in range(len(nei[i])):
                rx = min(abs(xt[nei[i][j]] - xx), L - abs(xt[nei[i][j]] - xx))
                ry = min(abs(yt[nei[i][j]] - yy), L - abs(yt[nei[i][j]] - yy))
                if np.sqrt(rx ** 2 + ry ** 2) < 5.8 * sigma:
                    xinner.append(xt[nei[i][j]])
                    yinner.append(yt[nei[i][j]])
                    tinner.append(thetat[nei[i][j]])
            if len(xinner) != 0:
                Er = calc_energy(len(xinner), xinner, yinner, tinner, xx, yy, tt0, T)
            else:
                Er = 0
            dE = Er - Er0

        elif len(nei[i]) == 0:
            dE = 0
        print('dE' + str(i) + '=', dE, 'dr[' + str(i) + ']=', np.sqrt(dx ** 2 + dy ** 2))
        if dE <= 0:
            xt[i] = xx
            yt[i] = yy
            delta_e += dE
            successr.append(1.0)
            attemptr.append(1.0)
        elif dE > 0:
            p = np.random.random()
            print(p, np.exp(- dE / (kb * T)))
            if p < np.exp(- dE / (kb * T)):
                xt[i] = xx
                yt[i] = yy
                delta_e += dE
                successr.append(1.0)
                attemptr.append(1.0)
            elif p >= np.exp(- dE / (kb * T)):
                xt[i] = xx0
                yt[i] = yy0
                successr.append(0.0)
                attemptr.append(1.0)

        xinnera = []
        yinnera = []
        tinnera = []

        if len(nei[i]) != 0:  # calculate potential energy from its neighbour molecules
            for j in range(len(nei[i])):
                rx = min(abs(xt[nei[i][j]] - xt[i]), L - abs(xt[nei[i][j]] - xt[i]))
                ry = min(abs(yt[nei[i][j]] - yt[i]), L - abs(yt[nei[i][j]] - yt[i]))
                if np.sqrt(rx ** 2 + ry ** 2) < 5.8 * sigma:
                    xinnera.append(xt[nei[i][j]])
                    yinnera.append(yt[nei[i][j]])
                    tinnera.append(thetat[nei[i][j]])
            if len(xinnera) != 0:
                Et = calc_energy(len(xinnera), xinnera, yinnera, tinnera, xt[i], yt[i], tt, T) \
                    - calc_energy(len(xinnera), xinnera, yinnera, tinnera, xt[i], yt[i], tt0, T)

            else:
                Et = 0
        elif len(nei[i]) == 0:
            Et = 0
        print('Et' + str(i) + '=', Et, 'dtheta[' + str(i) + ']=', dt)
        if Et <= 0:
            thetat[i] = tt
            delta_e += Et
            successt.append(1.0)
            attemptt.append(1.0)
        elif Et > 0:
            p = np.random.random()
            print(p, np.exp(- Et / (kb * T)))
            if p < np.exp(- Et / (kb * T)):
                thetat[i] = tt
                delta_e += Et
                successt.append(1.0)
                attemptt.append(1.0)
            elif p >= np.exp(- Et / (kb * T)):
                thetat[i] = tt0
                successt.append(0.0)
                attemptt.append(1.0)

        total_e01 = 0.0
        for k in range(NP):                       #### v4.2.1 update #### calculate energy after each molecule's move
            xin0 = []
            yin0 = []
            tin0 = []
            if len(nei[k]) != 0:  # calculate potential energy from its neighbour molecules
                for j in range(len(nei[k])):
                    rx0 = min(abs(xt[nei[k][j]] - xt[k]), L - abs(xt[nei[k][j]] - xt[k]))
                    ry0 = min(abs(yt[nei[k][j]] - yt[k]), L - abs(yt[nei[k][j]] - yt[k]))
                    if np.sqrt(rx0 ** 2 + ry0 ** 2) < 5.8 * sigma:
                        xin0.append(xt[nei[k][j]])
                        yin0.append(yt[nei[k][j]])
                        tin0.append(thetat[nei[k][j]])
                if len(xin0) != 0:
                    total_e01 += calc_energy(len(xin0), xin0, yin0, tin0, xt[k], yt[k], thetat[k], T) / 2.0
        print('TOTAL ENERGY 1 =', total_e01)


        i += 1
    #print(xt)

    dL = np.random.normal(scale=0.333) * abs(dl)
    ll = L + dL
    V = L ** 2
    VV = ll ** 2
    dV = VV - V
    f = ll / L
    dU = 0
    xtt = []
    ytt = []

    for i in range(NP):
        xtt.append(xt[i] * f)
        ytt.append(yt[i] * f)

    for i in range(NP):                                     # calculate potential energy of the system after deviation
        xi = []
        yi = []
        ti = []
        xin = []
        yin = []
        tin = []
        if len(nei[i]) != 0:                                # of length
            for j in range(len(nei[i])):                    # distances between molecules, after deviation of size
                rx = min(abs(xtt[nei[i][j]] - xtt[i]), L - abs(xtt[nei[i][j]] - xtt[i]))
                ry = min(abs(ytt[nei[i][j]] - ytt[i]), L - abs(ytt[nei[i][j]] - ytt[i]))
                if np.sqrt(rx ** 2 + ry ** 2) < 5.8 * sigma:
                    xi.append(xtt[nei[i][j]])
                    yi.append(ytt[nei[i][j]])
                    ti.append(thetat[nei[i][j]])
            for j in range(len(nei[i])):                    # distances between molecules, before deviation of size
                rx = min(abs(xt[nei[i][j]] - xt[i]), L - abs(xt[nei[i][j]] - xt[i]))
                ry = min(abs(yt[nei[i][j]] - yt[i]), L - abs(yt[nei[i][j]] - yt[i]))
                if np.sqrt(rx ** 2 + ry ** 2) < 5.8 * sigma:
                    xin.append(xt[nei[i][j]])
                    yin.append(yt[nei[i][j]])
                    tin.append(thetat[nei[i][j]])
            if len(xi) != 0:
                U = calc_energy(len(xi), xi, yi, ti, xtt[i], ytt[i], thetat[i], T)
            else:
                U = 0
            if len(xin) != 0:
                U0 = calc_energy(len(xin), xin, yin, tin, xt[i], yt[i], thetat[i], T)
            else:
                U0 = 0
            dU += (U - U0) / 2.0
        elif len(nei[i]) == 0:
            dU += 0
    print('dU =', dU, 'dL =', dL)
    dH = dU + P * dV - NP * kb * T * np.log(VV / V)
    print('dH =', dH)
#    print('dH = ', dH)
    A = []
    B = []
    if dH <= 0:
        L = ll
        for i in range(NP):
            A.append(xtt[i])
            B.append(ytt[i])
        xt = A
        yt = B
        delta_e += dU
        succ.append(1.0)
        atte.append(1.0)
    if dH > 0:
        p = np.random.random()
        print(p, np.exp(- dH / (kb * T)))
        if p <= np.exp(- dH / (kb * T)):
            L = ll
            for i in range(NP):
                A.append(xtt[i])
                B.append(ytt[i])
            xt = A
            yt = B
            delta_e += dU
            succ.append(1.0)
            atte.append(1.0)
        if p > np.exp(- dH / (kb * T)):
            succ.append(0.0)
            atte.append(1.0)
    #print(xt)
#    print(succ, atte)

    total_e01 = 0.0
    for k in range(NP):                 #### v4.2.1 update #### calculate energy after each molecule's move
        xin0 = []
        yin0 = []
        tin0 = []
        if len(nei[k]) != 0:  # calculate potential energy from its neighbour molecules
            for j in range(len(nei[k])):
                rx0 = min(abs(xt[nei[k][j]] - xt[k]), L - abs(xt[nei[k][j]] - xt[k]))
                ry0 = min(abs(yt[nei[k][j]] - yt[k]), L - abs(yt[nei[k][j]] - yt[k]))
                if np.sqrt(rx0 ** 2 + ry0 ** 2) < 5.8 * sigma:
                    xin0.append(xt[nei[k][j]])
                    yin0.append(yt[nei[k][j]])
                    tin0.append(thetat[nei[k][j]])
            if len(xin0) != 0:
                total_e01 += calc_energy(len(xin0), xin0, yin0, tin0, xt[k], yt[k], thetat[k], T) / 2.0
    print('TOTAL ENERGY 2 =', total_e01)

    total_energy.append(total_e)
    return xt, yt, thetat

def create_circle(x, y, l):
    circle = pl.Circle((x, y), radius = 0.5, color=l)
    return circle


def show_shape(patch):
    ax = pl.gca()
    ax.add_patch(patch)
    pl.axis('scaled')
    pl.xlim(0, L)
    pl.ylim(0, L)


def nematic_order(P):
    global successr, attemptr, successt, attemptt, succ, atte, L, nei, total_e, total_energy, delta_e, total_e01
#    np.random.seed(seed)
    sigma = 1
    T = 0.01
#    P = 10 ** 6
    NP = 100
    L = 100.0
    dr = np.zeros(NP)
    dtheta = np.zeros(NP)
    for i in range(NP):
        dr[i] = 0.01
        dtheta[i] = 0.001 * np.pi
    dl = 0.5
    step = 300
    X = np.zeros(7 * NP)                                        # every atom's position
    Y = np.zeros(7 * NP)
    Theta = []
    xstate = []
    ystate = []
    x, y, theta = mc_init(L, NP, T)                             # initializing
    z = []
    for i in range(NP):
        z.append(theta[i])                                      # write down every molecule's angle
    Theta.append(z)

    for i in range(NP):
        xss = []
        yss = []
        for j in range(7):
            xss.append(x[i] + (j - 3) * 0.8 * sigma * np.cos(theta[i]))
            yss.append(y[i] + (j - 3) * 0.8 * sigma * np.sin(theta[i]))
            if xss[j] > L:
                xss[j] -= L
            if xss[j] < 0:
                xss[j] += L
            if yss[j] > L:
                yss[j] -= L
            if yss[j] < 0:
                yss[j] += L
        for j in range(7):
            X[i + j * NP] = xss[j]
            Y[i + j * NP] = yss[j]
    A = []
    B = []
    for i in range(7 * NP):
        A.append(X[i])                                          # write down every molecule's position
        B.append(Y[i])
    xstate.append(A)
    ystate.append(B)

    Attemptr = []
    Successr = []
    Attemptt = []
    Successt = []
    i = 0
    sr = []
    st = []
    SR = []
    DR = []
    DTHETA = []
    DL = []
    STEP = []
    STEPS = []
    succ = []
    atte = []
    total_energy = []
    total_e = 0
    total_e01 = 0
    while i < step:
        attemptr = []
        successr = []
        attemptt = []
        successt = []
        print(i)
        if i % 20 == 0 and i > 0:  # adjust deviations every 20 steps before updating neighbour list
            SUC = 0.0
            ATT = 0.0
            for n in range(i - 20, i):
                SUC += succ[n]
                ATT += atte[n]
            SA = SUC / ATT
            SR.append(SA)
            STEP.append(i)
            print('SA = ', SA)
            if SA > 0.55:
                dl += 0.2 * abs(dl)
            if SA < 0.45:
                #                   if dl > 0.05:
                dl -= 0.2 * abs(dl)
                #                   if dl <= 0.05:
                #                       dl += 0.0
            if 0.45 <= SA <= 0.55:
                dl += 0.0

            for j in range(NP):
                sucr = 0.0
                attr = 0.0
                suct = 0.0
                attt = 0.0
                for n in range(i - 20, i):
                    sucr += Successr[n][j]
                    attr += Attemptr[n][j]
                    suct += Successt[n][j]
                    attt += Attemptt[n][j]
                sar = sucr / attr
                sat = suct / attt
                sr.append(sar)
                st.append(sat)
                STEPS.append(i)
                #                DR.append(dr)
                #                DTHETA.append(dtheta)
                #                DL.append(dl)


#                print('i = ', i, 'j = ', j, 'sar = ', sar, 'sat = ', sat)  # SA is mostly 1.0 or 0.0 ???????
                #                print(dr, dtheta, dl)
                if sar > 0.70:
                    dr[j] += 0.5 * abs(dr[j])

                if 0.60 < sar <= 0.70:
                    dr[j] += 0.2 * abs(dr[j])

                if 0.50 <= sar <= 0.60:
                    dr[j] += 0.1 * abs(dr[j])

                if 0.40 <= sar < 0.50:
                    #                    if dr >= 0.005:
                    dr[j] -= 0.1 * abs(dr[j])

                    #                    elif dr < 0.005:
                    #                    dr += 0.0
                    #                    dtheta += 0.0
                if 0.30 <= sar < 0.40:
                    #                   if dr >= 0.005:
                    dr[j] -= 0.2 * abs(dr[j])

                    #                   elif dr < 0.005:
                    #                       dr += 0.0
                    #                       dtheta += 0.0
                if sar < 0.30:
                    dr[j] -= 0.5 * abs(dr[j])

                if sat > 0.70:
                    dtheta[j] += 0.5 * abs(dtheta[j])
                if 0.60 < sat <= 0.70:
                    dtheta[j] += 0.2 * abs(dtheta[j])
                if 0.50 <= sat <= 0.60:
                    dtheta[j] += 0.1 * abs(dtheta[j])
                if 0.40 <= sat < 0.50:
                    #                    if dr >= 0.005:
                    dtheta[j] -= 0.1 * abs(dtheta[j])
                    #                    elif dr < 0.005:
                    #                    dr += 0.0
                    #                    dtheta += 0.0
                if 0.30 <= sat < 0.40:
                    #                   if dr >= 0.005:
                    dtheta[j] -= 0.2 * abs(dtheta[j])
                    #                   elif dr < 0.005:
                    #                       dr += 0.0
                    #                       dtheta += 0.0
                if sat < 0.30:
                    dtheta[j] -= 0.5 * abs(dtheta[j])
        print(dr, dtheta)
#        if i % 10 == 0:
#            random_molecule = randint(0, NP - 1)
#            random_end = randint(0, 1)
#            if random_end == 0:
#                x[random_molecule] -= sigma * np.cos(theta[random_molecule])
#                y[random_molecule] -= sigma * np.sin(theta[random_molecule])
#            if random_end == 1:
#                x[random_molecule] += sigma * np.cos(theta[random_molecule])
#                y[random_molecule] += sigma * np.sin(theta[random_molecule])

#            theta[random_molecule] += np.random.randn() * dtheta[random_molecule]

        if i % 10 == 0:                 # Update neighbour list every 10 steps
            nei = []                    # Create one at the start of update
            drm = max(dr)
            for h in range(NP):         # For each molecule, count neighbour
                neij = []               # molecules
                for j in range(NP):     # Mark neighbour molecules
                    if j != h:
                        rx = min(abs(x[h] - x[j]), L - abs(x[h] - x[j]))
                        ry = min(abs(y[h] - y[j]), L - abs(y[h] - y[j]))
                        r = np.sqrt(rx ** 2 + ry ** 2)
                        if r < 5.8 * sigma + 20 * np.sqrt(2) * (drm + dl):                 # ! latest update: 10 -> 20
                            neij.append(j)
                    else:
                        continue
                nei.append(neij)

#        elif i % 10 != 0:                # Update molecules' positions in the same
#            for g in range(NP):          # neighbour list
#                if len(nei[g]) != 0:
#                    for j in range(len(nei[g])):
#                        xnei[g][j] = x[nei[g][j]]
#                        ynei[g][j] = y[nei[g][j]]
#                        tnei[g][j] = theta[nei[g][j]]
#                elif len(nei[g]) == 0:
#                    continue
        total_e = 0.0
        for k in range(NP):                #### v4.2 update ####
            xin0 = []
            yin0 = []
            tin0 = []
            if len(nei[k]) != 0:  # calculate potential energy from its neighbour molecules
                for j in range(len(nei[k])):
                    rx0 = min(abs(x[nei[k][j]] - x[k]), L - abs(x[nei[k][j]] - x[k]))
                    ry0 = min(abs(y[nei[k][j]] - y[k]), L - abs(y[nei[k][j]] - y[k]))
                    if np.sqrt(rx0 ** 2 + ry0 ** 2) < 5.8 * sigma:
                        xin0.append(x[nei[k][j]])
                        yin0.append(y[nei[k][j]])
                        tin0.append(theta[nei[k][j]])
                if len(xin0) != 0:
                    total_e += calc_energy(len(xin0), xin0, yin0, tin0, x[k], y[k], theta[k], T) / 2.0
        print('TOTAL ENERGY  =', total_e)

        x00, y00, theta00 = mc_update(dl, dr, dtheta, x, y, theta, NP, T, P)
        print('delta energy =', delta_e)          #### v4.2 update ####
        Successr.append(successr)
        Attemptr.append(attemptr)
        Successt.append(successt)
        Attemptt.append(attemptt)

        x, y, theta = x00, y00, theta00
        s = []
        for l in range(NP):                         # write down every molecule's angle after every step
            s.append(theta[l])
        Theta.append(s)
        for m in range(NP):                         # write down every molecule's position after every step
            xss = []
            yss = []
            for j in range(7):
                xss.append(x[m] + (j - 3) * 0.8 * sigma * np.cos(theta[m]))
                yss.append(y[m] + (j - 3) * 0.8 * sigma * np.sin(theta[m]))
                if xss[j] > L:
                    xss[j] -= L
                if xss[j] < 0:
                    xss[j] += L
                if yss[j] > L:
                    yss[j] -= L
                if yss[j] < 0:
                    yss[j] += L
            for j in range(7):
                X[m + j * NP] = xss[j]
                Y[m + j * NP] = yss[j]
        A = []
        B = []
        for k in range(7 * NP):
            A.append(X[k])
            B.append(Y[k])
        pl.figure()
        pl.xlim(0, L)
        pl.ylim(0, L)
        for m in range(7 * NP - 1):
            c = create_circle(A[m], B[m], 'c')
            show_shape(c)
        d = create_circle(A[7 * NP - 1], B[7 * NP - 1], 'r')             # mark a specific molecule to track its position
        show_shape(d)
        pl.savefig('/home/wq39/project/programs/31aug/image' + str(i) + '.png')
        pl.close()
        xstate.append(A)
        ystate.append(B)
        i += 1

    Q = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])        # nematic order parameter calculation
    no = []
    Step = []
    for i in range(step):
        sumxx = 0
        sumxy = 0
        sumyy = 0
        for j in range(NP):
            sumxx += np.cos(Theta[i][j]) ** 2
            sumxy += np.cos(Theta[i][j]) * np.sin(Theta[i][j])
            sumyy += np.sin(Theta[i][j]) ** 2
        q00 = 1.5 * sumxx / NP - 0.5
        q01 = 1.5 * sumxy / NP
        q10 = 1.5 * sumxy / NP
        q11 = 1.5 * sumyy / NP - 0.5

        Q[0][0] = q00
        Q[0][1] = q01

        Q[1][0] = q10
        Q[1][1] = q11
        w, v = np.linalg.eig(Q)
        no.append(max(w[0], w[1], w[2]))
        Step.append(i)
    pl.figure()
#    pl.plot(Step, no)
#    pl.plot(STEPS, sr)
#    pl.plot(STEP, SR)
    pl.plot(Step, total_energy)
    pl.xscale('log')
    pl.yscale('log')
#    pl.plot(STEP, DR)
#    pl.plot(STEP, DTHETA)
#    pl.plot(STEP, DL)
    pl.savefig('/home/wq39/project/programs/31aug/Pressure' + str(P) + '.png')
    pl.close()
#    images = []
#    for i in range(100):
#        images.append(imageio.imread('/Users/weiyiqian/PycharmProjects/2DMCoverlap/NPTv4_2/image' + str(i) + '.png'))
#    imageio.mimsave('7disks_v4_5.mp4', images)
    return no


no = nematic_order(args.P)
#f = open('P' + str(100000) + '.txt', 'w')
#f.write('[')
#for s in no:
#    f.write(str(s) + ', ')
#f.write(']')
#f.close()










