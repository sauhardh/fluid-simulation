import glfw
from OpenGL.GL import *
import numpy as np

# -------------CONFIG
N = 64  # gird size
WINDOW_SIZE = 600
density = np.zeros((N, N), dtype=np.float32)

Vx = np.zeros((N, N), dtype=np.float32)
Vy = np.zeros((N, N), dtype=np.float32)


class Draw:
    def __init__(self):
        self.cell = 2.0 / N
        glBegin(GL_QUADS)

    def draw_density(self):
        for i in range(N):
            for j in range(N):
                d = density[i, j]
                glColor(d, d, d)

                x = -1 + i * self.cell
                y = -1 + j * self.cell

                glVertex2f(x, y)
                glVertex2f(x + self.cell, y)
                glVertex2f(x + self.cell, y + self.cell)
                glVertex2f(x, y + self.cell)
        glEnd()


def main():
    if not glfw.init():
        return

    window = glfw.create_window(
        WINDOW_SIZE, WINDOW_SIZE, "Fluid Simulation", None, None
    )

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Loop until user closes the window
    while not glfw.window_should_close(window):
        # Render here using pyopengl

        density[16, 16] = 1

        draw = Draw().draw_density()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
