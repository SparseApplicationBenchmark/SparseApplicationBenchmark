import math

# CONSTANTS
nsteps = 1000
savefreq = 10
density = 0.0005
mass = 0.01
cutoff = 0.01
min_r = cutoff / 100
dt = 0.0005


class Particle:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

    # def __repr__(self):
    #     # This tells Python how to represent the object as a string
    #     return (
    #         f"Particle(pos=({self.x}, {self.y}), "
    #         f"vel=({self.vx}, {self.vy}), "
    #         f"acc=({self.ax}, {self.ay}))"
    #     )

    def __eq__(self, other):
        if not isinstance(other, Particle):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.vx == other.vx
            and self.vy == other.vy
            and self.ax == other.ax
            and self.ay == other.ay
        )


def apply_force(particle, neighbor):
    dx = neighbor.x - particle.x
    dy = neighbor.y - particle.y
    r2 = dx * dx + dy * dy

    if r2 > cutoff * cutoff:
        return

    r2 = max(r2, min_r * min_r)
    r = math.sqrt(r2)

    coef = (1 - cutoff / r) / r2 / mass
    particle.ax += coef * dx
    particle.ay += coef * dy


def move(p, size):
    p.vx += p.ax * dt
    p.vy += p.ay * dt
    p.x += p.vx * dt
    p.y += p.vy * dt

    while p.x < 0 or p.x > size:
        if p.x < 0:
            p.x = -p.x
        else:
            p.x = 2 * size - p.x

        p.vx = -p.vx

    while p.y < 0 or p.y > size:
        if p.y < 0:
            p.y = -p.y
        else:
            p.y = 2 * size - p.y
        p.vy = -p.vy


def simulate_one_step(parts, num_parts, size):
    for i in range(num_parts):
        parts[i].ax = 0
        parts[i].ay = 0
        for j in range(num_parts):
            apply_force(parts[i], parts[j])

    for i in range(num_parts):
        move(parts[i], size)


def init_simulation(parts, num_parts, size, steps):
    for _ in range(steps):
        simulate_one_step(parts, num_parts, size)
