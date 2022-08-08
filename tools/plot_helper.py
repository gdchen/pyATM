"""
Created on Wed Jul  7 22:52:28 2021

@author: yaoyichen

>>>  available writers
>>> import matplotlib.animation as manimation
>>> manimation.writers.list()
['pillow', 'ffmpeg', 'ffmpeg_file', 'html']
"""


def generate_line_movement_gif(x, y, file_name, fps=50, xlim=(0, 1), ylim=(0, 1)):
    """
    x: x_axis of the data
    y:2d data, n_time x 1d_len
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set(xlim=xlim, ylim=ylim)
    line = ax.plot(x, y[0, :], color='k', lw=2)[0]

    def animate(i):
        line.set_ydata(y[i, :])
    anim = FuncAnimation(fig, animate, interval=0,
                         frames=len(y), repeat=True)
    fig.show()
    anim.save(file_name,
              fps=fps, dpi=80,  writer='pillow')


def generate_image_movement_gif(data, file_name, fps=40):
    """
    x: x_axis of the data
    y:2d data, n_time x 1d_len
    """
    print(data.shape)
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(data[0, :, :], aspect='auto',
                   cmap='jet', animated=True)

    fig.colorbar(im)

    print("min value:{}, max value:{}".format(np.min(data), np.max(data)))

    def init():
        im.set_array(data[0, :, :])
        im.set_clim(vmin=0.15*np.min(data), vmax=0.15 * np.max(data))
        return im,

    def update(i):
        im.set_array(data[i, :, :])
        print(i)
        return im,
    fig.show()
    anim = FuncAnimation(fig, update, frames=len(
        data), init_func=init, interval=0, blit=True)

    anim.save(file_name,
              fps=fps, dpi=80, writer='pillow')


def generate_image_movement3d_gif(data, grid_x, grid_y, file_name, fps=40):
    """
    https://pythonmatplotlibtips.blogspot.com/2018/01/rotate-elevation-angle-animation-3d-python-matplotlib-pyplot.html
    """

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np

    def update_plot(frame_number, zarray, plot):
        plot[0].remove()
        plot[0] = ax.plot_surface(
            grid_x, grid_y, zarray[frame_number, :, :], cmap="jet")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=5)

    plot = [ax.plot_surface(
        grid_x, grid_y, data[0, :, :], rstride=1, cstride=1)]

    ax.set_zlim(0.25*np.min(data),  0.25 * np.max(data))
    ani = FuncAnimation(
        fig, update_plot, len(data), fargs=(data, plot), interval=1000 / fps)

    ani.save(file_name, fps=fps, dpi=80, writer='pillow')


def generate_image_movement2d_gif(grid_x, grid_y, data,  file_name, fps=10):
    """
    https://pythonmatplotlibtips.blogspot.com/2018/01/rotate-elevation-angle-animation-3d-python-matplotlib-pyplot.html
    """

    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np

    def update_plot(frame_number, zarray, plot):

        print(frame_number)
        # Adjust the .001 to get finer gradient
        vmin = 0.25 * np.min(data)
        vmax = 0.25 * np.max(data)
        clev = np.arange(vmin, vmax, (vmax-vmin)/100.)
        plot[0] = ax.contourf(
            grid_y, grid_x,  zarray[frame_number, :, :], clev, cmap="jet", vmin=vmin, vmax=vmax)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmin = 0.25 * np.min(data)
    vmax = 0.25 * np.max(data)
    clev = np.arange(vmin, vmax, (vmax - vmin) / 100.)

    plot = [ax.contourf(
        grid_y, grid_x, data[0, :, :], clev, cmap="jet", vmin=vmin, vmax=vmax)]
    plt.axis('equal')

    # ax.set_zlim(,  )
    ani = FuncAnimation(
        fig, update_plot, len(data), fargs=(data, plot), interval=1000 / fps)

    ani.save(file_name, fps=fps, dpi=80, writer='pillow')
