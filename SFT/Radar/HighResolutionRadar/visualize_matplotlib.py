import numpy as np
import matplotlib.pyplot as plt


def rfft(x_data, y_data, fname=None):

    plt.figure()

    plt.plot(x_data, y_data)
    plt.title('Range-FFT')
    plt.xlabel('range (m)')
    plt.ylabel('power  (dB)')

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def rdm(ax_doppler, ax_range, data, fname=None, **fig_kw):

    plt.figure(**fig_kw)
    plt.contourf(ax_doppler, ax_range, np.transpose(data), cmap='viridis')
    plt.colorbar()
    plt.ylabel('range (m)')
    plt.xlabel('velocity (m/s)')

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def ram(ax_azimuth, ax_range, data, fname=None, **fig_kw):
    plt.figure(**fig_kw)
    plt.contourf(ax_azimuth, ax_range,  np.transpose(data), cmap='viridis')

    plt.colorbar()
    plt.xlabel('angle (°)')
    plt.ylabel('range (m)')

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def ram_polar(ax_azimuth, ax_range, data, fname=None, **fig_kw):

    r, theta = np.meshgrid(ax_range, np.radians(ax_azimuth))

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), **fig_kw)

    cax = ax.contourf(theta, r, data)
    ax.set_thetamin(np.degrees(np.min(theta)))
    ax.set_thetamax(np.degrees(np.max(theta)))
    ax.set_theta_zero_location("N")
    ax.set_title('Polar Plot')
    ax.set_xlabel('azimuth')
    ax.set_ylabel('range')
    plt.colorbar(cax, shrink=.5, pad=0.08)

    if fname is not None:
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()
