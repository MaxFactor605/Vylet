"""
Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math

import Box2D
import numpy as np

from gymnasium.error import DependencyNotInstalled

try:
    from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")


SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE # default 100000000 * self.size * self.size
WHEEL_MOMENT_OF_INERTIA = 1000 * SIZE * SIZE # default 4000 * self.size * self.size
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE # 1000000 * self.size * self.size
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27 # default 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


class Car:
    def __init__(self, world, init_angle, init_x, init_y, full_randomize = False, determined_randomize = False, randomize_percent = 0.5):
        self.world: Box2D.b2World = world
        if full_randomize:
            self.random_factor_engine = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_wheel = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_friction = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_brake = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_size = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_wheel_rad = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_wheel_width = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)

            self.random_factor_max_steer_step = np.random.uniform(low=1 - randomize_percent, high = 1 + randomize_percent)
            self.random_factor_max_steer_angle = np.random.uniform(low = 1 - randomize_percent, high = 1 + randomize_percent)
  
        elif determined_randomize:
            self.random_factor_engine = 1  + randomize_percent
            self.random_factor_wheel = 1 + randomize_percent
            self.random_factor_friction = 1 + randomize_percent
            self.random_factor_brake = 1 + randomize_percent
            self.random_factor_size = 1 + randomize_percent
            self.random_factor_wheel_rad = 1 + randomize_percent
            self.random_factor_wheel_width = 1 + randomize_percent
            self.random_factor_max_steer_step = 1 + randomize_percent
            self.random_factor_max_steer_angle = 1 + randomize_percent
        else:
            self.random_factor_engine = 1
            self.random_factor_wheel = 1
            self.random_factor_friction = 1
            self.random_factor_brake = 1
            self.random_factor_size = 1
            self.random_factor_wheel_rad = 1
            self.random_factor_wheel_width = 1
            self.random_factor_max_steer_step = 1
            self.random_factor_max_steer_angle = 1




        self.size = SIZE * self.random_factor_size
        self.wheel_r = WHEEL_R * self.random_factor_wheel_rad
        self.wheel_w = WHEEL_W * self.random_factor_wheel_width
        self.engine_power = 100000000 * self.size * self.size * self.random_factor_engine
        self.wheel_moment_of_inertia = 4000 * self.size * self.size * self.random_factor_wheel
        self.friction_limit = (
            1000000 * self.size * self.size * self.random_factor_friction
        ) 
       
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * self.size, y * self.size) for x, y in HULL_POLY4]
                    ),
                    density=1.0,
                ),
            ],
        )
        if full_randomize or determined_randomize:
            self.hull.color = np.random.uniform(low = 0, high = 1, size = 3)
        #elif determined_randomize:
        #    self.hull.color = (1/(1+np.exp(-randomize_percent)),1/(1+np.exp(-randomize_percent)) ,1/(1+np.exp(-randomize_percent)))
        else:
            self.hull.color = (0.8, 0, 0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-self.wheel_w, +self.wheel_r),
            (+self.wheel_w, +self.wheel_r),
            (+self.wheel_w, -self.wheel_r),
            (-self.wheel_w, -self.wheel_r),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * self.size, init_y + wy * self.size),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * self.size, y * front_k * self.size)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * self.wheel_r * self.size
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * self.size, wy * self.size),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * self.size * self.size ,
                motorSpeed=0,
                lowerAngle=-0.4 * self.random_factor_max_steer_angle,
                upperAngle=+0.4 * self.random_factor_max_steer_angle,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        """control: rear wheel drive

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation"""
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0 * self.random_factor_max_steer_step)

            # Position => friction_limit
            grass = True
            friction_limit = self.friction_limit * 1  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, self.friction_limit * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # self.wheel_moment_of_inertia*np.square(w.omega)/2 = E -- energy
            # self.wheel_moment_of_inertia*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/self.wheel_moment_of_inertia/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * self.engine_power
                * w.gas
                / self.wheel_moment_of_inertia
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * self.engine_power * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15 * self.random_factor_brake  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * self.size * self.size
            p_force *= 205000 * self.size * self.size
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if False:#abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / self.wheel_moment_of_inertia

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=2, closed=False
                )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                color = [int(c * 255) for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

                if "phase" not in obj.__dict__:
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-self.wheel_w* self.size, +self.wheel_r * c1 * self.size),
                    (+self.wheel_w* self.size, +self.wheel_r * c1 * self.size),
                    (+self.wheel_w* self.size, +self.wheel_r * c2 * self.size),
                    (-self.wheel_w* self.size, +self.wheel_r * c2 * self.size),
                ]
                white_poly = [trans * v for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

    def _create_particle(self, point1, point2, grass):
        class Particle:
            pass

        p = Particle()
        p.color = WHEEL_COLOR if not grass else MUD_COLOR
        p.ttl = 1
        p.poly = [(point1[0], point1[1]), (point2[0], point2[1])]
        p.grass = grass
        self.particles.append(p)
        while len(self.particles) > 30:
            self.particles.pop(0)
        return p

    def destroy(self):
        self.world.DestroyBody(self.hull)
        self.hull = None
        for w in self.wheels:
            self.world.DestroyBody(w)
        self.wheels = []
