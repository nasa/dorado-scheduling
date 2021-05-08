Slew time (`dorado.scheduling.slew_time`)
=========================================
.. autoclass:: dorado.scheduling.slew_time

.. plot::
    :include-source: False

    from matplotlib import pyplot as plt
    import numpy as np

    fig_width, fig_height = plt.rcParams['figure.figsize']
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(fig_width, 1.5 * fig_height))

    ax1.plot([0, 1, 3, 4], [1, 0, -1, -1], drawstyle='steps-post')
    ax2.plot([0, 1, 3, 4], [0, 1, 1, 0])

    t = np.arange(0, 4.1, 0.1)
    x = np.where(
        t <= 1,
        0.5 * np.square(t),
        np.where(
            t <= 3,
            0.5 + (t - 1),
            0.5 + 2 + 0.5 - 0.5 * np.square(4 - t)))
    ax3.plot(t, x)
    ax3.set_xlabel('Time')
    ax1.set_ylabel('Acceleration')
    ax2.set_ylabel('Velocity')
    ax3.set_ylabel('Distance')
    ax3.set_xticks([])
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels([r'$-a_\mathrm{max}$', '0', r'$+a_\mathrm{max}$'])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0', r'$v_\mathrm{max}$'])
    ax3.set_yticks([])
    fig.suptitle('Optimal slew')
