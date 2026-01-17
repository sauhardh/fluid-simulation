import glfw
from OpenGL.GL import *
import numpy as np
import colorsys
import time


class SmokyFluidSimulator:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height

        # Smoky fluid properties - higher diffusion, lower viscosity
        self.dt = 0.08
        self.diff = 0.0001  # Higher diffusion for smoke spread
        self.visc = 0.0000005  # Very low viscosity for wispy movement
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
        self.buoyancy = 0.05  # Gentler rise for smoke
        self.ambient_temp = 0.0

        # Mouse interaction
        self.mouse_down = False
        self.prev_mouse_x = 0
        self.prev_mouse_y = 0
        self.color_hue = 0.0

        self.time = 0.0

        # Pre-allocate arrays for vorticity
        self.curl = np.zeros((height, width), dtype=np.float32)
        self.abs_curl = np.zeros((height, width), dtype=np.float32)
        self.grad_x = np.zeros((height, width), dtype=np.float32)
        self.grad_y = np.zeros((height, width), dtype=np.float32)

    def add_density_colored(self, x, y, amount, r, g, b, radius=8):
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)

        jj, ii = np.meshgrid(np.arange(x_min, x_max) - x, np.arange(y_min, y_max) - y)
        dist = np.sqrt(ii**2 + jj**2)
        mask = dist < radius
        # Gaussian falloff for smoother smoke edges
        strength = np.where(mask, np.exp(-(dist**2) / (radius * 0.5) ** 2) * amount, 0)

        self.density_r[y_min:y_max, x_min:x_max] += strength * r
        self.density_g[y_min:y_max, x_min:x_max] += strength * g
        self.density_b[y_min:y_max, x_min:x_max] += strength * b
        self.temperature[y_min:y_max, x_min:x_max] += strength * 2

    def add_velocity(self, x, y, vx, vy):
        radius = 4
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius + 1)

        jj, ii = np.meshgrid(np.arange(x_min, x_max) - x, np.arange(y_min, y_max) - y)
        dist = np.sqrt(ii**2 + jj**2)
        mask = dist < radius
        strength = np.where(mask, np.exp(-(dist**2) / (radius * 0.4) ** 2), 0)

        self.u[y_min:y_max, x_min:x_max] += vx * strength
        self.v[y_min:y_max, x_min:x_max] += vy * strength

    def set_bounds(self, b, x):
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
        a = dt * diff * (self.width - 2) * (self.height - 2)
        c = 1 + 4 * a
        for _ in range(12):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1]
                + a * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
            ) / c
            self.set_bounds(b, x)

    def advect(self, b, d, d0, u, v, dt):
        dt0 = dt * (self.width - 2)

        j_grid, i_grid = np.meshgrid(
            np.arange(1, self.width - 1), np.arange(1, self.height - 1)
        )

        x = j_grid - dt0 * u[1:-1, 1:-1]
        y = i_grid - dt0 * v[1:-1, 1:-1]

        x = np.clip(x, 0.5, self.width - 1.5)
        y = np.clip(y, 0.5, self.height - 1.5)

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

        div[1:-1, 1:-1] = (
            -0.5 * h * (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1])
        )
        p[:] = 0

        self.set_bounds(0, div)
        self.set_bounds(0, p)

        for _ in range(12):
            p[1:-1, 1:-1] = (
                div[1:-1, 1:-1]
                + p[:-2, 1:-1]
                + p[2:, 1:-1]
                + p[1:-1, :-2]
                + p[1:-1, 2:]
            ) / 4
            self.set_bounds(0, p)

        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h

        self.set_bounds(1, u)
        self.set_bounds(2, v)

    def add_vorticity_confinement(self):
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

    def apply_buoyancy(self):
        temp_diff = self.temperature - self.ambient_temp
        self.v += temp_diff * self.buoyancy * self.dt
        self.temperature *= 0.995

    def add_turbulence(self):
        # Subtle random turbulence for organic smoke movement
        noise_strength = 0.05 if self.turbulence_mode else 0.02
        noise_u = (
            np.random.random((self.height, self.width)).astype(np.float32) - 0.5
        ) * noise_strength
        noise_v = (
            np.random.random((self.height, self.width)).astype(np.float32) - 0.5
        ) * noise_strength

        # Only add turbulence where there's smoke
        smoke_mask = (self.density_r + self.density_g + self.density_b) > 0.1
        self.u += noise_u * smoke_mask
        self.v += noise_v * smoke_mask

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

        # Slower fade for lingering smoke
        self.density_r *= 0.995
        self.density_g *= 0.995
        self.density_b *= 0.995

        # Velocity damping for smoother movement
        self.u *= 0.999
        self.v *= 0.999

    def get_texture_data(self):
        # Smoky color processing - softer, more translucent
        r = self.density_r
        g = self.density_g
        b = self.density_b

        # Apply gamma for softer smoke appearance
        gamma = 0.7
        r = np.power(np.clip(r, 0, 1), gamma)
        g = np.power(np.clip(g, 0, 1), gamma)
        b = np.power(np.clip(b, 0, 1), gamma)

        # Subtle glow in dense areas
        total = r + g + b
        glow = np.clip(total * 0.1, 0, 0.3)
        r += glow
        g += glow
        b += glow

        # Stack and convert
        rgb = np.stack([r, g, b], axis=2)
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        return rgb


def main():
    if not glfw.init():
        return

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)

    window_width, window_height = 800, 800

    window = glfw.create_window(
        window_width, window_height, "Smoky Fluid Simulator", None, None
    )
    if not window:
        glfw.terminate()
        return

    screen_width = mode.size.width
    screen_height = mode.size.height
    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2
    glfw.set_window_pos(window, window_x, window_y)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    sim = SmokyFluidSimulator(256, 256)

    def mouse_button_callback(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            sim.mouse_down = action == glfw.PRESS

    def cursor_position_callback(window, xpos, ypos):
        x = int(xpos / window_width * sim.width)
        y = int((window_height - ypos) / window_height * sim.height)

        if sim.mouse_down:
            dx = x - sim.prev_mouse_x
            dy = y - sim.prev_mouse_y

            # Smoky colors - muted, desaturated palette
            sim.color_hue = (sim.color_hue + 0.002) % 1.0

            # Lower saturation for smoke-like colors
            r, g, b = colorsys.hsv_to_rgb(sim.color_hue, 0.3, 0.9)

            # Add some gray to make it smokier
            gray = 0.6
            r = r * (1 - gray) + 0.8 * gray
            g = g * (1 - gray) + 0.8 * gray
            b = b * (1 - gray) + 0.8 * gray

            # FIX: Reduced sensitivity - smaller radius and amount
            # Original: sim.add_density_colored(x, y, 60, r, g, b, radius=12)
            sim.add_density_colored(x, y, 20, r, g, b, radius=6)

            # FIX: Gentler velocity for smoke-like movement
            # Original: sim.add_velocity(x, y, dx * 6, dy * 6)
            sim.add_velocity(x, y, dx * 3, dy * 3)

        sim.prev_mouse_x = x
        sim.prev_mouse_y = y

    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                # Capture image1.png
                data = sim.get_texture_data()
                # Need to flip vertically because OpenGL coords are bottom-up
                from PIL import Image
                img = Image.fromarray(np.flipud(data))
                img.save("image1.png")
                print("Snapshot saved as image1.png")
            elif key == glfw.KEY_2:
                # Capture image2.png
                data = sim.get_texture_data()
                from PIL import Image
                img = Image.fromarray(np.flipud(data))
                img.save("image2.png")
                print("Snapshot saved as image2.png")
            elif key == glfw.KEY_T:
                sim.turbulence_mode = not sim.turbulence_mode
                print(f"Turbulence Mode: {'ON' if sim.turbulence_mode else 'OFF'}")

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Initialize texture once
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        sim.width,
        sim.height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        None,
    )

    # Dark background for smoke visibility
    glClearColor(0.02, 0.02, 0.04, 1.0)

    print("=" * 60)
    print("SMOKY FLUID SIMULATOR")
    print("=" * 60)
    print("Controls:")
    print("  LEFT CLICK + DRAG: Create smoke & flow")
    print("  KEY '1' & '2'    : Take snapshots")
    print("  KEY 'T'          : Toggle Turbulence Mode")
    print("=" * 60)

    frame_count = 0
    fps_time = time.time()
    fps_counter = 0

    while not glfw.window_should_close(window):
        sim.step()

        # Ambient smoke puffs
        if frame_count % 120 == 0:
            xs = np.random.randint(30, sim.width - 30, 2)
            ys = np.random.randint(30, sim.height - 30, 2)
            hues = np.random.random(2)
            vels = (np.random.random((2, 2)) - 0.5) * 8

            for i in range(2):
                # Smoky gray-ish colors
                r, g, b = colorsys.hsv_to_rgb(hues[i], 0.2, 0.7)
                gray = 0.5
                r = r * (1 - gray) + 0.7 * gray
                g = g * (1 - gray) + 0.7 * gray
                b = b * (1 - gray) + 0.7 * gray

                sim.add_density_colored(xs[i], ys[i], 20, r, g, b, radius=10)
                sim.add_velocity(xs[i], ys[i], vels[i, 0], vels[i, 1])

        # Use SubImage for faster texture updates
        texture_data = sim.get_texture_data()
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            sim.width,
            sim.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            texture_data,
        )

        glClear(GL_COLOR_BUFFER_BIT)

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
