import sys
import platform
import matplotlib
import scipy
import numpy as np

from utils import printProgressBar


def solve_head_equation_singlevalue(dt, t, dx, x, bc, ic, k):
    print("")
    print("\t\t\tFD1D_HEAT_EXPLICIT_SINGLEVALUE_Ssk:")
    print("\t\t\t  Python version: %s" % (platform.python_version()))
    print("\t\t\t  Compute an approximate solution to the time-dependent")
    print("\t\t\t  one dimensional heat equation:")
    print("")
    print("\t\t\t    dH/dt - K * d2H/dx2 = f(x,t)")
    print("")

    cfl = k * dt / dx / dx  # This is the coefficient that determines convergence

    if 0.5 <= cfl:
        print("\t\t\tFD1D_HEAT_EXPLICIT_CFL - Fatal error!")
        print("\t\t\t  CFL condition failed.")
        print("\t\t\t  0.5 <= K * dT / dX / dX = %f" % (cfl))
        sys.exit(1)

    print("\t\t\t  Number of X nodes = %d" % (len(x)))
    print("\t\t\t  X interval is [%f,%f]" % (min(x), max(x)))
    print("\t\t\t  X spacing is %f" % (dx))
    print("\t\t\t  Number of T values = %d" % (len(t)))
    print("\t\t\t  T interval is [%f,%f]" % (min(t), max(t)))
    print("\t\t\t  T spacing is %f" % (dt))
    print("\t\t\t  Constant K = %g" % (k))
    print("\t\t\t  CFL coefficient = %g" % (cfl))

    hmat = np.zeros(
        (len(x), len(t))
    )  # running the code creates an output matrix of heads at each position and time

    for j in range(0, len(t)):
        if j % (int(len(t) / 20)) == 0:
            printProgressBar(j, len(t))
        if j == 0:
            h = ic * np.ones(len(x))
            h[0] = bc[0, 0]
            h[-1] = bc[1, 0]
        else:
            h_new = np.zeros(len(x))

            for c in range(1, len(x) - 1):
                l = c - 1
                r = c + 1
                h_new[c] = h[c] + cfl * (h[l] - 2.0 * h[c] + h[r])
            h_new[0] = bc[0, j]
            h_new[-1] = bc[1, j]
            h = h_new
        for i in range(0, len(x)):
            hmat[i, j] = h[i]
    print(" ")
    print("\t\t\t SOLVER COMPLETE")
    return hmat


def solve_head_equation_elasticinelastic(
    dt,
    t,
    dx,
    x,
    bc,
    ic,
    k_elastic,
    k_inelastic,
    overburdenstress=False,
    overburden_data=[],
    initial_precons=False,
    initial_condition_precons=[],
):
    print("")
    print("\t\t\tFD1D_HEAT_EXPLICIT_ELASTICINELASTIC_Ssk:")
    print("\t\t\t  Python version: %s" % (platform.python_version()))
    print("\t\t\t  Compute an approximate solution to the time-dependent")
    print("\t\t\t  one dimensional heat equation:")
    print("")
    print("\t\t\t    dH/dt - K * d2H/dx2 = f(x,t)")
    print("")
    if overburdenstress:
        print(" \t\t\tSOLVING WITH OVERBURDEN STRESS INCLUDED;  ")
        print("\t\t\tOverburden data read in as ")
        print(overburden_data)
    else:
        overburden_data = np.zeros_like(t)

    h_precons = np.zeros((len(x), len(t)))
    if not initial_precons:
        h_precons[:, 0] = ic
    else:
        print(
            "\t\t\t preset preconsolidation stress found to be",
            initial_condition_precons,
        )
        h_precons[:, 0] = initial_condition_precons
    inelastic_flag = np.zeros((len(x), len(t)))

    cfl_elastic = (
        k_elastic * dt / dx / dx
    )  # This is the coefficient that determines convergence
    cfl_inelastic = (
        k_inelastic * dt / dx / dx
    )  # This is the coefficient that determines convergence

    print("\t\t\t  Number of X nodes = %d" % (len(x)))
    print("\t\t\t  X interval is [%f,%f]" % (min(x), max(x)))
    print("\t\t\t  X spacing is %f" % (dx))
    print("\t\t\t  Number of T values = %d" % (len(t)))
    print("\t\t\t  T interval is [%f,%f]" % (min(t), max(t)))
    print("\t\t\t  T spacing is %f" % (dt))
    print("\t\t\t  Elastic K = %g" % (k_elastic))
    print("\t\t\t  Inelastic K = %g" % (k_inelastic))
    print("\t\t\t  CFL elastic coefficient = %g" % (cfl_elastic))
    print("\t\t\t  CFL inelastic coefficient = %g" % (cfl_inelastic))

    if 0.5 <= cfl_elastic:
        print("\t\t\tFD1D_HEAT_EXPLICIT_CFL - Fatal error!")
        print("\t\t\t  CFL condition failed.")
        print("\t\t\t  0.5 <= K * dT / dX / dX = %f" % max(cfl_elastic, cfl_inelastic))
        sys.exit(1)

    hmat = np.zeros(
        (len(x), len(t))
    )  # running the code creates an output matrix of heads at each position and time

    for j in range(0, len(t)):
        if j % (int(len(t) / 20)) == 0:
            printProgressBar(j, len(t))
        if j == 0:
            h = ic * np.ones(len(x))
            h[0] = bc[0, 0]
            h[-1] = bc[1, 0]
        else:
            h_new = np.zeros(len(x))

            for c in range(1, len(x) - 1):
                l = c - 1
                r = c + 1
                if inelastic_flag[c, j - 1]:
                    h_new[c] = (
                        h[c]
                        + cfl_inelastic * (h[l] - 2.0 * h[c] + h[r])
                        + (overburden_data[j] - overburden_data[j - 1])
                    )
                    # print(dt * overburden_data[j])
                #                    h_new[c] = h[c] + cfl * ( h[l] - 2.0 * h[c] + h[r] ) + dt * f[c] This is the forcing version

                elif not inelastic_flag[c, j - 1]:
                    h_new[c] = (
                        h[c]
                        + cfl_elastic * (h[l] - 2.0 * h[c] + h[r])
                        + (overburden_data[j] - overburden_data[j - 1])
                    )
                    # print(dt * overburden_data[j])

                else:
                    print("Uh oh! Neither inelastic or elastic..something went wrong.")

            h_new[0] = bc[0, j]
            h_new[-1] = bc[1, j]
            h = h_new
        for i in range(0, len(x)):
            hmat[i, j] = h[i]
            if h[i] - overburden_data[j] < h_precons[i, j]:
                if j <= len(t) - 3:
                    h_precons[i, j + 1] = h[i] - overburden_data[j]
                    inelastic_flag[i, j] = 1
            else:
                if j <= len(t) - 3:
                    h_precons[i, j + 1] = h_precons[i, j]
            if j == len(t) - 3:
                if h[i] - overburden_data[j] < h_precons[i, j]:
                    inelastic_flag[i, j] = 1

    #        if j <= len(t)-3:
    #            h_precons[:,j+1] = np.min(hmat[:,:j+1],axis=1)

    print(" ")
    print("\t\t\t SOLVER COMPLETE")
    return hmat, inelastic_flag


def subsidence_solver_aquitard_elasticinelastic(
    hmat,
    Sske,
    Sskv,
    dz,
    TESTn=1,
    overburden=False,
    unconfined=False,
    overburden_data=0,
    debuglevel=0,
    endnodes=False,
    preset_precons=False,
    ic_precons=[],
):
    ### TESTn is a temporary variable, referring to the number of midpoints done. If you start with 20 nodes and TESTn=1, you integrate over 20 nodes. If TESTn=2 you intergrate over 40 nodes, and so on. It can be used to reduce error from using the Riemann sum.
    print(
        "Running subsidence solver. Overburden=%s, unconfined=%s."
        % (overburden, unconfined)
    )
    if overburden:
        if not unconfined:
            print(" \t\t\tSOLVING WITH OVERBURDEN STRESS INCLUDED;  ")
            print("\t\t\tOverburden data read in as ")
            print(overburden_data)
        else:
            print(" \t\t\tSOLVING WITH OVERBURDEN STRESS INCLUDED;  ")
            print("\t\t\tThis aquifer is unconfined. ")
            print(overburden_data)

    print(
        "Aquitard solver is done at midpoints. Applying linear interpolation to hmat."
    )
    if not endnodes:
        hmat_interp = np.zeros(
            (np.shape(hmat)[0] * (2 * TESTn) - (2 * TESTn - 1), np.shape(hmat)[1])
        )
        for i in range(np.shape(hmat)[1]):
            if i % (int(np.shape(hmat)[1] / 20)) == 0:
                printProgressBar(i, np.shape(hmat)[1])
            if len(hmat[:, i]) != len(
                0.001 * np.arange(0, 1000 * np.shape(hmat)[0] * dz, 1000 * dz)
            ):
                print(
                    "ERROR: hmat is not the same length as 0.001*np.arange(0,1000*np.shape(hmat)[0]*dz,1000*dz). If dz_clays is not a multiple of the layer thickness, you may need to give it to more significant figures for this to work."
                )
                print(
                    0.001
                    * np.arange(
                        0,
                        1000 * (np.shape(hmat_interp)[0] + 1) * (dz / (2 * TESTn))
                        - 0.00001,
                        1000 * dz / (2 * TESTn),
                    )
                )
                sys.exit(1)
            a = scipy.interpolate.interp1d(
                0.001 * np.arange(0, 1000 * np.shape(hmat)[0] * dz, 1000 * dz),
                hmat[:, i],
                kind="linear",
            )
            #            print(np.arange(0,np.shape(hmat)[0]*dz,dz))
            ##            print(np.shape(hmat_interp)[0])
            ##            print(dz)
            ##            print(np.shape(a))
            #            print(np.arange(0,np.shape(hmat_interp)[0]*(dz/2),(dz/2)))
            hmat_interp[:, i] = a(
                0.001
                * np.arange(
                    0,
                    1000 * ((np.shape(hmat)[0] - 1) * dz) + 1,
                    1000 * dz / (2 * TESTn),
                )
            )  # again the 1000 and 0.001 is to ensure against np.arange's bad rounding with non integers
        if TESTn != 1:
            hmat_midpoints = hmat_interp[1:-1, :]
        else:
            hmat_midpoints = hmat_interp[1::2, :]
    else:
        print("DEBUG: not using midpoints for head solver.")
        hmat_midpoints = hmat
    # hmat_midpoints_precons = np.array([np.min(hmat_midpoints[:,:i+1],axis=1) for i in range(np.shape(hmat_midpoints)[1])]).T

    if overburden:
        overburden_data_midpoints = np.tile(
            overburden_data, (np.shape(hmat_midpoints)[0], 1)
        )
    else:
        overburden_data_midpoints = np.zeros_like(hmat_midpoints)

    stress_midpoints = overburden_data_midpoints - hmat_midpoints

    stress_midpoints_precons = np.zeros_like(hmat_midpoints)
    inelastic_flag_midpoints = np.zeros_like(hmat_midpoints)
    if preset_precons:
        print("preset precons found. interpolating to midpoints.")
        print("starting with", ic_precons)
        a = scipy.interpolate.interp1d(
            0.001 * np.arange(0, 1000 * np.shape(hmat)[0] * dz, 1000 * dz), ic_precons
        )
        ic_precons_interp = a(
            0.001
            * np.arange(
                0, 1000 * ((np.shape(hmat)[0] - 1) * dz) + 1, 1000 * dz / (2 * TESTn)
            )
        )
        ic_precons_initial = ic_precons_interp[1::2]
        print("interpoolated to", ic_precons_initial)
        stress_midpoints_precons[:, 0] = ic_precons_initial
    else:
        stress_midpoints_precons[:, 0] = (
            overburden_data_midpoints[:, 0] - hmat_midpoints[:, 0]
        )

    for i in range(np.shape(stress_midpoints)[1] - 1):
        if i % (int((np.shape(stress_midpoints)[1] - 1) / 20)) == 0:
            printProgressBar(i, np.shape(stress_midpoints)[1])
        for j in range(np.shape(stress_midpoints)[0]):
            if stress_midpoints[j, i] > stress_midpoints_precons[j, i]:
                stress_midpoints_precons[j, i + 1] = stress_midpoints[j, i]
                inelastic_flag_midpoints[j, i] = 1
            else:
                stress_midpoints_precons[j, i + 1] = stress_midpoints_precons[j, i]
                inelastic_flag_midpoints[j, i] = 0

    if debuglevel == 1:
        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(stress_midpoints, aspect="auto")
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.title("Stress at midpoints")

        matplotlib.pyplot.figure()
        matplotlib.pyplot.imshow(inelastic_flag_midpoints, aspect="auto")
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.title("Inelastic flag at midpoints")

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(stress_midpoints[10, :], "-", label="stress")
        matplotlib.pyplot.plot(hmat_midpoints[10, :], "-", label="head")
        matplotlib.pyplot.legend()
        ax2 = matplotlib.pyplot.twinx(matplotlib.pyplot.gca())
        ax2.plot(inelastic_flag_midpoints[10, :], "k--", label="inelastic flag")
        matplotlib.pyplot.title("stress head and inelastic flag at node 10")
        matplotlib.pyplot.show()

    inelastic_flag_midpoints = np.array(inelastic_flag_midpoints, dtype=bool)

    if TESTn != 1:
        dz = dz / (2 * TESTn)

    # print('doing db')
    # db = [dz*( (inelastic_flag_midpoints[:,i] * Sskv * (stress_midpoints[:,i+1] - stress_midpoints[:,i])) + ~inelastic_flag_midpoints[:,i] * Sske * (stress_midpoints[:,i+1] - stress_midpoints[:,i])) for i in range(np.shape(stress_midpoints)[1]-1)]
    # print('doing ds')
    # ds = [dz*( np.dot(inelastic_flag_midpoints[:,i] * Sskv, stress_midpoints[:,i+1] - stress_midpoints[:,i]) + np.dot(~inelastic_flag_midpoints[:,i] * Sske, stress_midpoints[:,i+1] - stress_midpoints[:,i])) for i in range(np.shape(stress_midpoints)[1]-1)]
    # print('doing ds elastic')
    # ds_elastic = [dz*(np.dot(~inelastic_flag_midpoints[:,i] * Sske, stress_midpoints[:,i+1] - stress_midpoints[:,i])) for i in range(np.shape(stress_midpoints)[1]-1)]
    # print('doing ds inelastic')
    # ds_inelastic = [dz*( np.dot(inelastic_flag_midpoints[:,i] * Sskv, stress_midpoints[:,i+1] - stress_midpoints[:,i]))  for i in range(np.shape(stress_midpoints)[1]-1)]

    # s = np.zeros(np.shape(hmat)[1])
    # s_elastic = np.zeros(np.shape(hmat)[1])
    # s_inelastic = np.zeros(np.shape(hmat)[1])

    # print('\tIntegrating deformation over time.')
    # for i in range(1,np.shape(hmat)[1]):
    #     if i % (int(np.shape(hmat)[1]/20)) == 0:
    #         printProgressBar(i,np.shape(hmat)[1]-1)
    #     s[i] = s[i-1]-ds[i-1]
    #     s_elastic[i] = s_elastic[i-1]-ds_elastic[i-1]
    #     s_inelastic[i] = s_inelastic[i-1]-ds_inelastic[i-1]

    b = np.zeros(np.shape(hmat)[1])
    for ti in range(1, np.shape(hmat)[1]):
        if ti % (int(np.shape(hmat)[1] / 20)) == 0:
            printProgressBar(ti, np.shape(hmat)[1] - 1)
        b[ti] = dz * (
            Sskv
            * np.sum(stress_midpoints_precons[:, ti] - stress_midpoints_precons[:, 0])
            - Sske * np.sum(stress_midpoints_precons[:, ti] - stress_midpoints[:, ti])
        )

    b = -1 * np.array(b)

    #    return db,s,s_elastic,s_inelastic,inelastic_flag_midpoints
    return b, inelastic_flag_midpoints
