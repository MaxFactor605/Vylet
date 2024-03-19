__credits__ = ["Andrea PIERRÉ"]

import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second 50
ZOOM = 2.7 # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += (1000.0 / len(self.env.track)) # Here control reward for each tile
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ### Action Space
    If continuous:
        There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking.
    If discrete:
        There are 5 actions: do nothing, steer left, steer right, gas, brake.

    ### Observation Space
    State consists of 96x96 pixels.

    ### Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ### Starting State
    The car starts at rest in the center of the road.

    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ### Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ### Reset Arguments
    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.
    Example usage:
    ```py
        env = gym.make("CarRacing-v1", domain_randomize=True)

        # normal reset, this changes the colour scheme by default
        env.reset()

        # reset with colour scheme change
        env.reset(options={"randomize": True})

        # reset with no colour scheme change
        env.reset(options={"randomize": False})
    ```

    ### Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ### Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = False,
        full_randomize: bool = False,
        determined_randomize: bool = True,
        checkpoints: int = 3,
        noise: float = 0.8,
        rad: float = 140, 
        right: bool = False,
        custom_continuous: bool = False,
        random_trase: bool = True,
        car_randomize_factor: float = 0,
        draw_trace: bool = False
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.custom_continuous = custom_continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()
        self.checkpoints = checkpoints
        self.noise = noise
        self.rad = rad
        self.right = right
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.random_trase = random_trase
        self.car_randomize_factor = car_randomize_factor
        self.path = []
        self.draw_trace = draw_trace
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([0, -1, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(80, STATE_W, 3), dtype=np.uint8  # STATE_H => 80
        )

        self.render_mode = render_mode
        self.action_mult = 1
        self.prev_action = 0
        self.car_steer_angle = 0
        self.base_step = 0.1
        self.full_randomize = full_randomize
        self.determined_randomize = determined_randomize
        self.step_count = 0
        self.rms_acc = 0
        self.prev_h = 0

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = self.checkpoints
        if self.random_trase:
            CHECKPOINTS = np.random.randint(2, 20)
        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            if self.random_trase:
                noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            else:
                noise = self.noise
            #noise =  math.pi * 1 / CHECKPOINTS
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            if self.random_trase:
                rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD) # TRACK_RAD = 900
            else:
                rad = self.rad
            #rad = TRACK_RAD-100
            if self.random_trase:
                if c == 0:
                    alpha = 0
                    rad = 1.5 * TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * math.pi * c / CHECKPOINTS
                    self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * TRACK_RAD
            else:
                if c == CHECKPOINTS - 1:
                    self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        self.angles = [0.0] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.25
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
            
            #self.angles[i] = abs(beta1 - beta2)
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            #if border[i] or (i > 2 and border[i - 2]) or (i < len(track)-2 and border[i+2]):
                #print(beta1, beta2)
                #print(abs(beta1 - beta2))
            #    t.color = np.clip(self.road_color + 255, 0 ,255)
            #else:
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
           # print([road1_l, road1_r, road2_r, road2_l])
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.border = border
        self.track = track
        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.inactive_mult = 0
        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        #if self.train_randomize and self.continuous:
            #self.base_step = np.random.uniform(low = 0.01, high = 0.1)
            #print("New base step: {}".format(self.base_step))
        #print(self.track[0][1], self.track[int(len(self.track)/2)][1])
        if self.right:
            self.car = Car(self.world, (math.pi + self.track[10][1]), *self.track[10][2:4], full_randomize = self.full_randomize, determined_randomize = self.determined_randomize, randomize_percent = self.car_randomize_factor)
        else:
            self.car = Car(self.world,  *self.track[0][1:4], full_randomize = self.full_randomize, determined_randomize = self.determined_randomize, randomize_percent = self.car_randomize_factor)
        self.prev_x, self.prev_y = self.car.hull.position
        
       # self.car.hull.angularVelocity = 0
        self.car_steer_angle = 0
        self.action_mult = 1
        self.prev_action = 0
        self.prev_angular_velocity = 0
        self.step_count = 0
        self.rms_acc = 0
        self.prev_h = 0
        self.path = []
        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        self.step_count += 1
        assert self.car is not None
        if action is not None:
            
            if self.continuous:
                self.car.steer(action[1])
                self.car.gas(action[0])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
            
                if action == self.prev_action and self.custom_continuous:
                    self.action_mult += 1
                else:
                    self.action_mult = 1

                #print(self.action_mult)
                
                
                if self.custom_continuous:
                    self.car_steer_angle = (-self.base_step * (action == 1) + self.base_step * (action == 2)) * self.action_mult
                    #print(self.car_steer_angle)
                    self.car_steer_angle = np.clip(self.car_steer_angle, -1, 1)
                    self.car.steer(self.car_steer_angle)
                    self.car.gas((action == 3) * self.action_mult * self.base_step)
                    self.car.brake((action == 4) * self.action_mult * self.base_step)
                else:
                    if action == 1:
                        self.car.steer(-1)
                    elif action == 2:
                        self.car.steer(1)
                    else:
                        self.car.steer(0)
                    #self.car.steer(#self.car_steer_angle)
                    self.car.gas((action == 3))
                    self.car.brake((action == 4))
        
        self.prev_action = action

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
            
                done = True
            x, y = self.car.hull.position
            
            
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

            min_dist = [PLAYFIELD, 0, 0] # nearest tile
            min_dist2 = [PLAYFIELD, 0, 0] # second nearest tile
            nearest_tile_id = 0
            for i, tile in enumerate(self.track):
                dist = math.sqrt((x - tile[2]) ** 2 + (y - tile[3]) ** 2)
                if(dist < min_dist[0]):
                    min_dist = [dist, tile[2], tile[3]]
                    nearest_tile_id = i
                elif dist < min_dist2[0]:
                    min_dist2 = [dist, tile[2], tile[3]]
            
            # Computing distance from road center
            y_ = min_dist[0]
            z_ = min_dist2[0]
            x_ = math.sqrt((min_dist[1] - min_dist2[1]) ** 2 + (min_dist[2] - min_dist2[2]) ** 2)
            cos_y = (x_**2+z_**2-y_**2)/(2*x_*z_)
            sin_y = math.sqrt(1-cos_y**2)
            h = z_*sin_y
            
            is_turn = False
            #turn_angle = 0
            if self.right:
                if nearest_tile_id - 10 < 0:
                    is_turn = sum(self.border[:nearest_tile_id+2]) > 0
                    #turn_angle = sum(self.angles[:nearest_tile_id+2])
                else:
                    if nearest_tile_id + 2 >= len(self.track):
                        is_turn = sum(self.border[nearest_tile_id-10:]) > 0
                    #    turn_angle = sum(self.angles[nearest_tile_id-5:])
                    else:
                        is_turn = sum(self.border[nearest_tile_id-10:nearest_tile_id+2]) > 0
                    #    turn_angle = sum(self.angles[nearest_tile_id-5:nearest_tile_id+2])
            else:
                if nearest_tile_id - 2 < 0:
                    is_turn = sum(self.border[:nearest_tile_id+10]) > 0
                    #turn_angle = sum(self.angles[:nearest_tile_id+2])
                else:
                    if nearest_tile_id + 10 >= len(self.track):
                        is_turn = sum(self.border[nearest_tile_id-2:]) > 0
                    #    turn_angle = sum(self.angles[nearest_tile_id-5:])
                    else:
                        is_turn = sum(self.border[nearest_tile_id-2:nearest_tile_id+10]) > 0
                    #    turn_angle = sum(self.angles[nearest_tile_id-5:nearest_tile_id+2])
            #print(turn_angle, self.angles[nearest_tile_id])

            
            #if nearest_tile_id + 5 >= len(self.track):
            #    is_turn = sum(self.border[nearest_tile_id-5:]) > 0
            #else:
            #    if nearest_tile_id - 5 < 0:
            #        is_turn = sum(self.border[:nearest_tile_id+5]) > 0
            #    else:
            #        is_turn = sum(self.border[nearest_tile_id-5:nearest_tile_id+5]) > 0            
            if self.random_trase:
                if self.t > 1.5:
                    step_reward += min(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 35, 0)/50
                if not is_turn:
                    if abs(self.prev_angular_velocity - self.car.hull.angularVelocity) >= 0.4:
                        step_reward -= 1
            else:
                if is_turn:
                    if self.checkpoints > 4:
                        step_reward -= max(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 65, 0)/10
                    elif self.checkpoints > 2:
                        step_reward -= max(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 55, 0)/10
                    else:
                        step_reward -= max(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 45, 0)/10
                    self.rms_acc = 0
                    self.step_count = 0
                else:
                    if self.t > 1:
                        step_reward -= self.car.wheels[0].joint.angle**2 * 10
                    #self.rms_acc += (h-self.prev_h)**2
                    #step_reward -= math.sqrt(self.rms_acc/self.step_count)
                    #print(math.sqrt(self.rms_acc/self.step_count) )
                    #print((h-self.prev_h)**2)
                    step_reward -= max(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 85, 0)/10
                    if self.t > 1.5:
                        step_reward += min(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2) - 35, 0)/20
                    if abs(self.prev_angular_velocity - self.car.hull.angularVelocity) >= 0.4:
                        step_reward -= 1

            if np.sign(self.car.hull.angularVelocity) != np.sign(self.prev_angular_velocity):
                step_reward -= 1  


            step_reward -= max(abs(self.car.hull.angularVelocity) - 4, 0)*3
            step_reward -= np.clip(h/TRACK_WIDTH, 0.05, 1)
            #print(h/10, step_reward)
            #if np.sign(self.car.hull.angularVelocity) != np.sign(self.prev_angular_velocity):
            #    print("uuuuu", self.t)
            if(h > 1.5*TRACK_WIDTH):
                step_reward -= 500
                done = True
            #print("X: {} Y: {} prev_X: {} prev_Y: {} h: {}".format(x, y, self.prev_x, self.prev_y, h))
            # Penalize if car position doesn't change
            if (math.fabs(x - self.prev_x) < 0.1 and math.fabs(y - self.prev_y) < 0.1):
                self.inactive_mult += 1
            else:
                self.inactive_mult = 0
            #print("{:.2f}\t{:.2f}\t{:.2f}".format(math.sqrt(self.car.hull.linearVelocity[0]**2 + self.car.hull.linearVelocity[1]**2), self.car.hull.angularVelocity, step_reward))
            #print(self.car.hull.angularDamping)
            #step_reward -= self.inactive_mult
            if self.inactive_mult > 150:
                step_reward -= 500
                done = True
            self.prev_x = x
            self.prev_y = y
            self.prev_angular_velocity = self.car.hull.angularVelocity
            self.prev_h = h
            if self.draw_trace:
                self.path.append([(x,y), (x+1, y) ,(x+1,y+1), (x,y+1)])
            #print(x,y)
        #print(self.tile_visited_count/len(self.track))
        if self.render_mode == "human":
            self.render()
        #print(step_reward)
        return self.state[:80], step_reward, done, done, {}

    def render(self, render_mode = None):
        if self.render_mode is None and render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        else:
            return self._render(self.render_mode if render_mode is None else render_mode)

    def take_screenshot(self):
        return self._render("rgb_array", 0.3)
    
    def _render(self, mode: str, zoom = ZOOM):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle if mode != "rgb_array" else 0
        # Animating first second zoom.
        zoom = 0.1 * SCALE * 0 + zoom * SCALE * 1
        scroll_x = -(self.car.hull.position[0]) * zoom if mode != "rgb_array" else 0
        scroll_y = -(self.car.hull.position[1]) * zoom if mode != "rgb_array" else 0
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1]) if mode != "rgb_array" else (WINDOW_W/2, WINDOW_H/2)

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        
        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )
       # print(translation)
        #path = [
        #    (c[0]*zoom, c[1] * zoom) for c in self.path
        #]
        #for (x,y) in self.path:
        #    gfxdraw.aapolygon(self.surf, [(x,y), (x,y+1), (x+1,y), (x+1,y+1)], [255,0,0])

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            #print(poly)
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)
        
        if self.draw_trace:
            for poly in self.path:
                poly = [(p[0], p[1]) for p in poly]
                self._draw_colored_polygon(self.surf, poly, [255,255, 255], zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    a = 0

    def register_input():
        global quit, restart, a
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a = 2
                elif event.key == pygame.K_RIGHT:
                    a = 1
                elif event.key == pygame.K_UP:
                    a = 3
                elif event.key == pygame.K_DOWN:
                    a = 4  # set 1.0 for wheels to block to zero rotation
                elif event.key == pygame.K_RETURN:
                    restart = True
                elif event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a = 0
                if event.key == pygame.K_RIGHT:
                    a = 0
                if event.key == pygame.K_UP:
                    a = 0
                if event.key == pygame.K_DOWN:
                    a = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human", continuous=False, custom_continuous=True)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str(a))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
