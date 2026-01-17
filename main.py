import glfw
from OpenGL.GL import *
import numpy as np
import colorsys
import time


class OptimizedFluidSimulator:
    def __init__(self, width=256, height=256):  # Reduced for better FPS
        self.width = width
        self.height = height

        # Optimized fluid properties
        self.dt = 0.1
        self.diff = 0.00001
        self.visc = 0.000001
        self.vorticity_strength = 0.4  # Increased for more natural turbulent eddies
        self.turbulence_mode = False

        # Velocity fields
        self.u = np.zeros((height, width), dtype=np.float32)
        self.v = np.zeros((height, width), dtype=np.float32)
        self.u_prev = np.zeros((height, width), dtype=np.float32)
        self.v_prev = np.zeros((height, width), dtype=np.float32)

        # Multiple density layers for color mixing
        self.density_r = np.zeros((height, width), dtype=np.float32)
        self.density_g = np.zeros((height, width), dtype=np.float32)
        self.density_b = np.zeros((height, width), dtype=np.float32)
        self.density_r_prev = np.zeros((height, width), dtype=np.float32)
        self.density_g_prev = np.zeros((height, width), dtype=np.float32)
        self.density_b_prev = np.zeros((height, width), dtype=np.float32)

        # Temperature field for buoyancy
        self.temperature = np.zeros((height, width), dtype=np.float32)
        self.temperature_prev = np.zeros((height, width), dtype=np.float32)
        self.buoyancy = 0.15
        self.ambient_temp = 0.0

        # Mouse interaction
        self.mouse_down = False
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.color_hue = 0.0

        self.time = 0.0

        # Pre-allocate arrays for vorticity (optimization)
        self.curl = np.zeros((height, width), dtype=np.float32)
        self.abs_curl = np.zeros((height, width), dtype=np.float32)
        self.grad_x = np.zeros((height, width), dtype=np.float32)
        self.grad_y = np.zeros((height, width), dtype=np.float32)

    def add_density_colored(self, x, y, amount, r, g, b):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.density_r[y, x] += amount * r
            self.density_g[y, x] += amount * g
            self.density_b[y, x] += amount * b
            self.temperature[y, x] += amount * 5

    def add_velocity(self, x, y, vx, vy):
        # Optimized brush - smaller radius for speed
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 2.5:
                        strength = 1.0 - dist / 2.5
                        self.u[ny, nx] += vx * strength
                        self.v[ny, nx] += vy * strength

    def set_bounds(self, b, x):
        # Vectorized boundary conditions for speed
        if b == 1:
            x[:, 0] = -x[:, 1]
            x[:, -1] = -x[:, -2]
        else:
            x[:, 0] = x[:, 1]
            x[:, -1] = x[:, -2]

        if b == 2:
            x[0, :] = -x[1, :]
            x[-1, :] = -x[-2, :]
        else:
            x[0, :] = x[1, :]
            x[-1, :] = x[-2, :]

        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    def diffuse(self, b, x, x0, diff, dt):
        # Reduced iterations for better FPS (20 is good balance)
        a = dt * diff * (self.width - 2) * (self.height - 2)
        for _ in range(20):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1]
                + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
            ) / (1 + 4 * a)
            self.set_bounds(b, x)

    def advect(self, b, d, d0, u, v, dt):
        # Fully vectorized advection (much faster!)
        dt0 = dt * (self.width - 2)

        # Create coordinate grids
        j_grid, i_grid = np.meshgrid(
            np.arange(1, self.width - 1), np.arange(1, self.height - 1)
        )

        # Trace backwards
        x = j_grid - dt0 * u[1:-1, 1:-1]
        y = i_grid - dt0 * v[1:-1, 1:-1]

        # Clamp to bounds
        x = np.clip(x, 0.5, self.width - 1.5)
        y = np.clip(y, 0.5, self.height - 1.5)

        # Bilinear interpolation
        i0 = y.astype(int)
        j0 = x.astype(int)
        i1 = i0 + 1
        j1 = j0 + 1

        s1 = x - j0
        s0 = 1 - s1
        t1 = y - i0
        t0 = 1 - t1

        d[1:-1, 1:-1] = t0 * (s0 * d0[i0, j0] + s1 * d0[i0, j1]) + t1 * (
            s0 * d0[i1, j0] + s1 * d0[i1, j1]
        )

        self.set_bounds(b, d)

    def project(self, u, v, p, div):
        h = 1.0 / self.width

        # Vectorized divergence calculation
        div[1:-1, 1:-1] = (
            -0.5 * h * (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1])
        )
        p[:] = 0

        self.set_bounds(0, div)
        self.set_bounds(0, p)

        # Reduced iterations for speed
        for _ in range(20):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1]
                + p[:-2, 1:-1]
                + p[2:, 1:-1]
                + p[1:-1, :-2]
                + p[1:-1, 2:]
            ) / 4
            self.set_bounds(0, p)

        # Subtract pressure gradient
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h

        self.set_bounds(1, u)
        self.set_bounds(2, v)

    def add_vorticity_confinement(self):
        # Fully vectorized vorticity calculation
        self.curl[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2])
            - (self.u[2:, 1:-1] - self.u[:-2, 1:-1])
        ) * 0.5

        self.abs_curl[:] = np.abs(self.curl)

        self.grad_x[1:-1, 1:-1] = (
            self.abs_curl[1:-1, 2:] - self.abs_curl[1:-1, :-2]
        ) * 0.5
        self.grad_y[1:-1, 1:-1] = (
            self.abs_curl[2:, 1:-1] - self.abs_curl[:-2, 1:-1]
        ) * 0.5

        magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2) + 1e-5
        self.grad_x /= magnitude
        self.grad_y /= magnitude

        self.u += self.vorticity_strength * self.grad_y * self.curl * self.dt
        self.v += -self.vorticity_strength * self.grad_x * self.curl * self.dt

    def add_turbulence(self):
        # Noise strength depends on toggle
        noise_strength = 0.05 if self.turbulence_mode else 0.01
        noise_u = (
            np.random.random((self.height, self.width)).astype(np.float32) - 0.5
        ) * noise_strength
        noise_v = (
            np.random.random((self.height, self.width)).astype(np.float32) - 0.5
        ) * noise_strength

        # Add turbulence where there is density
        mask = (self.density_r + self.density_g + self.density_b) > 0.1
        self.u += noise_u * mask
        self.v += noise_v * mask

    def apply_buoyancy(self):
        temp_diff = self.temperature - self.ambient_temp
        self.v += temp_diff * self.buoyancy * self.dt
        self.temperature *= 0.99

    def step(self):
        self.time += self.dt

        self.apply_buoyancy()
        self.add_turbulence()

        # Velocity step
        self.diffuse(1, self.u_prev, self.u, self.visc, self.dt)
        self.diffuse(2, self.v_prev, self.v, self.visc, self.dt)

        self.project(self.u_prev, self.v_prev, self.u, self.v)

        self.advect(1, self.u, self.u_prev, self.u_prev, self.v_prev, self.dt)
        self.advect(2, self.v, self.v_prev, self.u_prev, self.v_prev, self.dt)

        self.add_vorticity_confinement()

        self.project(self.u, self.v, self.u_prev, self.v_prev)

        # Density step for each color
        for density, density_prev in [
            (self.density_r, self.density_r_prev),
            (self.density_g, self.density_g_prev),
            (self.density_b, self.density_b_prev),
        ]:
            self.diffuse(0, density_prev, density, self.diff, self.dt)
            self.advect(0, density, density_prev, self.u, self.v, self.dt)

        # Temperature advection
        self.diffuse(0, self.temperature_prev, self.temperature, self.diff * 2, self.dt)
        self.advect(0, self.temperature, self.temperature_prev, self.u, self.v, self.dt)

        # Fade
        self.density_r *= 0.998
        self.density_g *= 0.998
        self.density_b *= 0.998

    def get_texture_data(self):
        # Vectorized color processing
        r = np.clip(self.density_r * 1.5, 0, 1)
        g = np.clip(self.density_g * 1.5, 0, 1)
        b = np.clip(self.density_b * 1.5, 0, 1)

        # Glow effect
        total = r + g + b
        glow_mask = total > 1.0
        excess = np.maximum(total - 1.0, 0)
        r += excess * 0.3
        g += excess * 0.3
        b += excess * 0.3

        # Stack and convert in one operation
        rgb = np.stack([r, g, b], axis=2)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        return rgb


def main():
    if not glfw.init():
        return

    # Get primary monitor for centering
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)

    window_width, window_height = 800, 800

    # Create window
    window = glfw.create_window(
        window_width, window_height, "Optimized Fluid Simulator", None, None
    )
    if not window:
        glfw.terminate()
        return

    # CENTER THE WINDOW!
    screen_width = mode.size.width
    screen_height = mode.size.height
    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2
    glfw.set_window_pos(window, window_x, window_y)

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # VSync

    # Smaller grid for better performance
    sim = OptimizedFluidSimulator(256, 256)

    def mouse_button_callback(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            sim.mouse_down = action == glfw.PRESS

    def cursor_position_callback(window, xpos, ypos):
        x = int(xpos / window_width * sim.width)
        y = int((window_height - ypos) / window_height * sim.height)

        if sim.mouse_down:
            dx = x - sim.prev_mouse_x
            dy = y - sim.prev_mouse_y

            # Rainbow colors
            sim.color_hue = (sim.color_hue + 0.005) % 1.0
            r, g, b = colorsys.hsv_to_rgb(sim.color_hue, 0.9, 1.0)

            # Optimized brush size
            for i in range(-3, 4):
                for j in range(-3, 4):
                    dist = np.sqrt(i * i + j * j)
                    if dist < 3:
                        strength = (1.0 - dist / 3.0) * 30
                        sim.add_density_colored(x + j, y + i, strength, r, g, b)

            # Add velocity
            sim.add_velocity(x, y, dx * 6, dy * 6)

        sim.prev_mouse_x = x
        sim.prev_mouse_y = y

    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                data = sim.get_texture_data()
                from PIL import Image
                img = Image.fromarray(np.flipud(data))
                img.save("image1_main.png")
                print("Snapshot saved as image1_main.png")
            elif key == glfw.KEY_2:
                data = sim.get_texture_data()
                from PIL import Image
                img = Image.fromarray(np.flipud(data))
                img.save("image2_main.png")
                print("Snapshot saved as image2_main.png")
            elif key == glfw.KEY_T:
                sim.turbulence_mode = not sim.turbulence_mode
                print(f"Turbulence Mode: {'ON' if sim.turbulence_mode else 'OFF'}")

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)

    # OpenGL setup with better blending
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glClearColor(0.0, 0.0, 0.05, 1.0)

    print("=" * 60)
    print("OPTIMIZED FLUID SIMULATOR - SMOOTH 60 FPS")
    print("=" * 60)
    print("Controls:")
    print("  LEFT CLICK + DRAG: Create fluid & flow")
    print("  KEY '1' & '2'    : Take snapshots")
    print("  KEY 'T'          : Toggle Turbulence Mode")
    print("=" * 60)

    # FPS counter
    frame_count = 0
    fps_time = time.time()
    fps_counter = 0

    while not glfw.window_should_close(window):
        # Update simulation
        sim.step()

        # Ambient motion (less frequent for performance)
        if frame_count % 90 == 0:
            for _ in range(2):
                x = np.random.randint(30, sim.width - 30)
                y = np.random.randint(30, sim.height - 30)
                hue = np.random.random()
                r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.8)
                sim.add_density_colored(x, y, 25, r, g, b)
                vx = (np.random.random() - 0.5) * 15
                vy = (np.random.random() - 0.5) * 15
                sim.add_velocity(x, y, vx, vy)

        # Update texture
        texture_data = sim.get_texture_data()
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            sim.width,
            sim.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            texture_data,
        )

        glClear(GL_COLOR_BUFFER_BIT)

        # Draw quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()

        glfw.swap_buffers(window)
        glfw.poll_events()

        # FPS counter
        frame_count += 1
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_time >= 1.0:
            print(f"FPS: {fps_counter}")
            fps_counter = 0
            fps_time = current_time

    glfw.terminate()


if __name__ == "__main__":
    main()
