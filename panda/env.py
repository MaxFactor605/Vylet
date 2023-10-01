# Textures: https://www.sketchuptextureclub.com/textures/architecture/roads/roads/road-texture-pack-seamless-07628

import sys
#import direct.directbase.DirectStart
import yaml
import math
import time
import numpy as np
import gymnasium as gym
import os

from gymnasium import spaces
from PIL import Image

from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState
from direct.showbase.ShowBase import ShowBase

from panda3d.core import *
from panda3d.direct import *
from direct.showbase.DirectObject import DirectObject
from direct.task import Task
#from panda3d.graphics import GraphicsOutput



from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletDebugNode
from panda3d.bullet import BulletVehicle
from panda3d.bullet import ZUp


TILES_DIR = "./tiles"
MAPS_DIR = "./maps"
MAP_MULT = 1.5
TILE_WIDTH = 10
TILE_LENGTH = 10
TILE_HEIGTH = 2
CAMERA_HEIGHT = 2
CAMERA_ANGLE = 70
RENDER_CAMERA_FOV = 40
CAR_CAMERA_FOV = 40

OBS_SIZE = 128

M_MAT = np.array([[1, -2, 1],
                  [-2, 2, 0],
                  [1,  0, 0]])


class Tile():
    def __init__(self, name, node_path, is_drivable, bazier_curves, x, y):
        self.name = name
        self.node_path = node_path
        self.is_drivable = is_drivable
        self.bazier_curves = bazier_curves
        self.x = x
        self.y = y
        self.visited = 0


    def set_visited(self, val):
        if self.is_drivable:
            self.visited = val

class MyEnv(gym.Env):
    def __init__(self, render_mode = None, view_mode = "first-person", map_file = "./test_map.yaml", frame_skip = 1, max_n_steps = 1024, light_rand = True):
        self.start_tile = None
        self.num_steps = 0
        self.render_mode = render_mode
        self.view_mode = view_mode
        self.frame_skip = frame_skip
        self.max_n_steps = max_n_steps
        self.tiles_visited = 0
        self.complete_percent = 0.0
        self.lightRandomization = light_rand
        self.current_tile = None
        self.current_curve = None

        if render_mode == "human":
            self.base = ShowBase(windowType="onscreen")
            self.base.cam.node().getLens().setFov(RENDER_CAMERA_FOV)
        else:
            self.base = ShowBase(windowType="offscreen")

        self.observation_space = spaces.Box(low = 0, high = 255, shape = [OBS_SIZE,OBS_SIZE,3], dtype = np.uint8)

        self.action_space = spaces.Box(np.array([0, -1, 0]), np.array([+1,+1,+1]))
        
        self.addOffScreenRender()
        #self.debugNP.hide()

        self.addWorld()

        self.addLight()

        self.generate_tiles(map_file)

        self.addVehicle()
        if render_mode == "human":
            if view_mode == 'up-down':
                self.base.cam.setPos(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, (self.map_width*(TILE_WIDTH*MAP_MULT)/2)/math.tan((RENDER_CAMERA_FOV/720)*math.pi)/2)
                #print(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_width, )
                self.base.cam.lookAt(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, 0)
            elif view_mode == 'back':
                self.base.cam.setPos(self.map_width*(TILE_WIDTH*MAP_MULT)/2, -50, 20)
                self.base.cam.lookAt(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, 0)
            elif view_mode == 'back-follow':
                self.base.cam.reparentTo(self.VehicleNP)
                self.base.cam.setPos(0.5, -10, 3)
                self.base.cam.lookAt(0, 0, 1)
            elif view_mode == "first-person":
                self.base.cam.reparentTo(self.VehicleNP)
                self.base.cam.setPos(0,0,CAMERA_HEIGHT)
                self.base.cam.lookAt(0,CAMERA_HEIGHT * math.tan((CAMERA_ANGLE/360)*(2*math.pi)),0)
        if self.start_tile is None:
            raise Exception("Start not specified")
        
        #self.base.taskMgr.add(self.update, 'updateWorld')

       # self.accept('escape', self.doExit)
       # self.accept('r', self.doReset)
       # self.accept('f1', self.toggleWireframe)
       # self.accept('f2', self.toggleTexture)
       # self.accept('f3', self.toggleDebug)
       # self.accept('f5', self.doScreenshot)
      
            

        
    def addLight(self, reset = False):
        if self.lightRandomization:
            color_factor = np.random.uniform(low = 0.05, high = 2)
            temp_factor = np.random.uniform(low = 0.05, high = 3)
        else:
            color_factor = 1
            temp_factor = 1
        l = 0.7 * color_factor
        t = 6500 * temp_factor
        
       
        self.alight = AmbientLight('ambientLight')
        self.alight.setColor(Vec4(l, l, l, 1))
        self.alight.setColorTemperature(t)
      
        self.alightNP = self.base.render.attachNewNode(self.alight)

       
        self.dlight = DirectionalLight('directionalLight')
        #dlight.setDirection(Vec3(1, 1, 10))
        self.dlight.setColor(Vec4(l, l, l, 1))
        self.dlight.setColorTemperature(t)
      
        self.dlightNP = self.base.render.attachNewNode(self.dlight)
        self.dlightNP.setHpr(0, -60, 0)

        self.base.render.clearLight()
        self.base.render.setLight(self.alightNP)
        self.base.render.setLight(self.dlightNP)

    def addWorld(self):
        self.worldNP = self.base.render.attachNewNode('World')
        
        self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
        #self.debugNP.show()

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(self.debugNP.node())
        
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)

        self.ground = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.ground.node().addShape(shape)
        self.ground.setPos(0, 0, 3)
        self.ground.setCollideMask(BitMask32.allOn())

        self.world.attachRigidBody(self.ground.node())


    def addOffScreenRender(self):
        fb_prop = FrameBufferProperties()
        fb_prop.setRgbColor(True)
        # Only render RGB with 8 bit for each channel, no alpha channel
        fb_prop.setRgbaBits(8, 8, 8, 0)
        fb_prop.setDepthBits(24)

        # Create window properties
        win_prop = WindowProperties.size(OBS_SIZE, OBS_SIZE)

        # Create window (offscreen)
        window = self.base.graphicsEngine.makeOutput(self.base.pipe, "cameraview", 0, fb_prop, win_prop, GraphicsPipe.BFRefuseWindow)
        #lens = PerspectiveLens()
        #lens.set_fov(45)  # Set the field of view to 90 degrees
        #lens.set_near(1.0)  # Set the near clipping plane
        #lens.set_far(100.0)  # Set the far clipping plane
        cam_obj = Camera("Car_camera")
        cam_obj.set_scene(self.base.render)
        self.CarCamNp = NodePath(cam_obj)
        # Create display region
        # This is the actual region used where the image will be rendered
        disp_region = window.makeDisplayRegion()
        disp_region.setCamera(self.CarCamNp)

        # Create the texture where the frame will be rendered
        # This is the RGB/RGBA buffer which stores the rendered data
        self.bgr_tex = Texture()
        window.addRenderTexture(self.bgr_tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

    def addVehicle(self):

        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0, 0, 0.5))

        self.VehicleNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Vehicle'))
        self.VehicleNP.node().addShape(shape, ts)
        self.VehicleNP.setPos(self.start_tile.x + TILE_LENGTH/2*MAP_MULT, self.start_tile.y  + TILE_WIDTH/2 * MAP_MULT, 3)
        self.VehicleNP.node().setMass(800.0)
        self.VehicleNP.node().setDeactivationEnabled(False)

        self.world.attachRigidBody(self.VehicleNP.node())


        # Vehicle
        self.vehicle = BulletVehicle(self.world, self.VehicleNP.node())
        self.vehicle.setCoordinateSystem(ZUp)
        self.world.attachVehicle(self.vehicle)
       

        self.yugoNP = self.base.loader.loadModel('bullet-samples/models/yugo/yugo.egg')
        self.yugoNP.reparentTo(self.VehicleNP)
        self.CarCamNp.reparentTo(self.VehicleNP)
        self.CarCamNp.setPos(0.0, 0.0, CAMERA_HEIGHT)
        self.CarCamNp.lookAt(0, CAMERA_HEIGHT * math.tan((CAMERA_ANGLE/360)*(2*math.pi)), 0)
        self.CarCamNp.node().getLens().setFov(CAR_CAMERA_FOV)

        # Right front wheel
        np = self.base.loader.loadModel('bullet-samples/models/yugo/yugotireR.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70,  1.05, 0.3), True, np)

        # Left front wheel
        np = self.base.loader.loadModel('bullet-samples/models/yugo/yugotireL.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70,  1.05, 0.3), True, np)

        # Right rear wheel
        np = self.base.loader.loadModel('bullet-samples/models/yugo/yugotireR.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3( 0.70, -1.05, 0.3), False, np)

        # Left rear wheel
        np = self.base.loader.loadModel('bullet-samples/models/yugo/yugotireL.egg')
        np.reparentTo(self.worldNP)
        self.addWheel(Point3(-0.70, -1.05, 0.3), False, np)

        # Steering info
        self.steering = 0.0            # degree
        self.steeringClamp = 45.0      # degree
        self.steeringIncrement = 120.0 # degree per second

        self.maxEngineForce = 1000
        self.maxBrakeForce = 100

        self.vehicle_starting_transform = self.VehicleNP.get_transform()

    def addWheel(self, pos, front, np):
        wheel = self.vehicle.createWheel()

        wheel.setNode(np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)

        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(0.25)
        wheel.setMaxSuspensionTravelCm(40.0)

        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(100.0);
        wheel.setRollInfluence(0.1)

    def generate_tiles(self, map_filename = None):
        self.mapNP = NodePath("Map")
        self.mapNP.reparentTo(self.ground)
        assert map_filename is not None
        
        with open(map_filename, "r") as file:
            map_yaml = yaml.safe_load(file)
        
        self.map_width = len(map_yaml["tiles"][0])
        self.map_length = len(map_yaml["tiles"])
        self.path_length = map_yaml["path_length"]
        y = (self.map_length-1) * TILE_LENGTH * MAP_MULT
        
        self.tiles = []
         
        for row in map_yaml["tiles"]:
            if len(row) != self.map_width:
                raise IndexError
            x = 0
            row_tiles = []
            for tile_name in row:
                if "/" in tile_name:
                    tile_name, specs = tile_name.split("/")
                else:
                    specs = None
                tile = self.base.loader.loadModel(TILES_DIR + "/{}".format(tile_name) + "/plane_{}.glb".format(tile_name))
                tile.reparentTo(self.mapNP)
                # Apply scale and position transforms on the model.
                #tile.setHpr(tile, Vec3(0, 90, 0))

                tile.setScale(MAP_MULT, MAP_MULT, 1)
                tile.setPos(x, y,-1)

                with open(TILES_DIR + "/{}/conf.yaml".format(tile_name), "r") as file:
                    conf_yaml = yaml.safe_load(file)

                is_drivable = conf_yaml["is_drivable"]

                if is_drivable and "curves" in conf_yaml.keys():
                    curves = [np.array(curve) for curve in conf_yaml["curves"]]
                else:
                    curves = None

                row_tiles.append(Tile(tile_name, tile, is_drivable, curves, x, y)) 
                if specs is not None:
                    if specs == "start":
                        self.start_tile = row_tiles[-1]
                        #print(self.start_tile.x, self.start_tile.y)
                x += TILE_WIDTH * MAP_MULT
            self.tiles = [row_tiles, *self.tiles]
            y -= TILE_LENGTH * MAP_MULT

        #for y in self.tiles:
        #    for x in y:
        #        print(x.name, end = " ")
        #    print("\n")
        #sys.exit()

                
    @staticmethod
    def get_t_vector(t):
        return np.array([t**2, t, 1])

    
    def get_distance_and_dir(self, curve, point_x, point_y, pos_vec):
        #curve_points = []
        min_dist = 9999
        for t in np.arange(0.0, 1.01, 0.01):
            point_on_curve = np.matmul(np.matmul(curve, M_MAT), self.get_t_vector(t))
            #print(point_on_curve)
            dist = math.sqrt((point_x-point_on_curve[0]) ** 2 + (point_y - point_on_curve[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                if t == 1.0:
                    closest_point2 = point_on_curve
                    closest_point1 = np.matmul(np.matmul(curve, M_MAT), self.get_t_vector(t - 0.1))
                else:
                    closest_point1 = point_on_curve
                    closest_point2 = np.matmul(np.matmul(curve, M_MAT), self.get_t_vector(t + 0.1))
        

        curve_dir = closest_point2 - closest_point1
        curve_dir /= math.sqrt(curve_dir[0] ** 2 + curve_dir[1] ** 2)
        car_dir = np.array([pos_vec[0], pos_vec[1]])

        dir = np.dot(curve_dir, car_dir)
        dir = np.clip(dir, -1.0, +1.0)
        return min_dist, dir
    
    def computeReward(self, current_tile, car_x, car_y, tile_change):
        if(current_tile.bazier_curves is not None):
            max_dir = -2
            current_curve_t = None
            actual_dist = 0
            for curve in current_tile.bazier_curves:
                curve = curve.copy()
                curve[0] *= TILE_LENGTH*MAP_MULT; curve[1] *= TILE_WIDTH*MAP_MULT
                curve[0] += current_tile.x; curve[1] += current_tile.y
                dist, dir = self.get_distance_and_dir(curve, car_x, car_y, self.vehicle.forward_vector)
                if (dir > max_dir):
                    max_dir = dir
                    current_curve_t = curve
                    actual_dist = dist

            reward = -actual_dist/1.5 + max_dir/2
            if self.num_steps > 100:
                reward += min(self.vehicle.getCurrentSpeedKmHour() - 20, 0)/5
            if not tile_change and (current_curve_t != self.current_curve).any():
                reward -= 100
                #print("Curve changed!")
            self.current_curve = current_curve_t
           # print("{:.3f}, {:.3f}".format(actual_dist, max_dir))
        else:
            raise Exception("Current tile is drivable but curves not found")
   
        return reward
    def reset(self, seed = None, map_file = None):
        if not map_file is None:
            self.mapNP.removeNode()
            self.generate_tiles(map_file)
        else:
            for row in self.tiles:
                for tile in row:    
                    if tile.is_drivable:
                        tile.set_visited(0)

        if self.lightRandomization:
            if np.random.random() < 0.5:
                color_factor = np.random.uniform(low = 1/5, high = 1)
                l = 0.5 * color_factor
                self.alight.setColor(Vec4(l, l, l, 1))
                self.dlight.setColor(Vec4(l, l, l, 1))
            else:
                temp_factor = np.random.uniform(low = 0.1, high = 3)
                t = 6500 * temp_factor
                self.alight.setColorTemperature(t)
                self.dlight.setColorTemperature(t)
            color_factor = 1
            temp_factor = 1
        self.VehicleNP.set_transform(self.vehicle_starting_transform)
        zeroVector = Vec3(0,0,0)
        self.VehicleNP.setPos(self.start_tile.x + TILE_LENGTH/2*MAP_MULT, self.start_tile.y  + TILE_WIDTH/2 * MAP_MULT, 3)
        self.steering = 0.0
        self.num_steps = 0
        self.tiles_visited = 0
        self.complete_percent = 0.0
        self.vehicle.applyEngineForce(0, 0);
        self.vehicle.applyEngineForce(0, 1);
        self.vehicle.setBrake(0, 0);
        self.vehicle.setBrake(0, 1);
        self.vehicle.setSteeringValue(self.steering, 0);
        self.vehicle.setSteeringValue(self.steering, 1);
        for wheel in self.vehicle.getWheels():
            wheel.setRotation(0.0)
        self.vehicle.chassis.setLinearVelocity(zeroVector)
        self.vehicle.chassis.setAngularVelocity(zeroVector)
        self.vehicle.chassis.clearForces()
        if self.render_mode == "human":
            if self.view_mode == 'up-down':
                self.base.cam.setPos(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, (self.map_width*(TILE_WIDTH*MAP_MULT)/2)/math.tan((RENDER_CAMERA_FOV/720)*math.pi)/2)
                #print(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_width, )
                self.base.cam.lookAt(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, 0)
            elif self.view_mode == 'back':
                self.base.cam.setPos(self.map_width*(TILE_WIDTH*MAP_MULT)/2, -50, 20)
                self.base.cam.lookAt(self.map_width*(TILE_WIDTH*MAP_MULT)/2, self.map_length*(TILE_LENGTH*MAP_MULT)/2, 0)
        
        return self.step(), {}


    def step(self, action = None):
        self.base.taskMgr.step()
        dt = 0.02 #globalClock.getDt()
       # print(dt)
        done = False
        reward = 0
        steering_sign = np.sign(self.steering)
        if action is not None:
            self.processInput(dt, action)
            self.world.doPhysics(dt, 10, 0.008)
            if self.frame_skip is not None:
                for i in range(self.frame_skip):
                    self.base.taskMgr.step()
                    dt = globalClock.getDt()
                    self.world.doPhysics(dt, 10, 0.008)
            car_x, car_y, car_z = self.yugoNP.getPos(self.worldNP)
            if self.render_mode == "human":
                if self.view_mode == "up-down-follow":
                    self.base.cam.setPos(car_x, car_y, 30)
                    self.base.cam.lookAt(car_x, car_y, 0)
            current_tile_t = self.tiles[int(car_y//(TILE_LENGTH*MAP_MULT))][int(car_x//(TILE_WIDTH*MAP_MULT))]
            tile_change = current_tile_t != self.current_tile
            self.current_tile = current_tile_t
            #print(current_tile.name, self.vehicle.forward_vector)
            if not self.current_tile.is_drivable:
                reward = -500
                done = True
            elif (abs(self.vehicle.forward_vector[2])) > 0.1:
                reward = -500
                done = True
            else:
                reward = self.computeReward(self.current_tile, car_x, car_y, tile_change)
                if np.sign(self.steering) != steering_sign:
                    reward -= 1
                if not self.current_tile.visited:
                    self.tiles_visited += 1
                    self.complete_percent = self.tiles_visited/self.path_length
                    self.current_tile.set_visited(1)
                    reward += 20
                if self.num_steps == self.max_n_steps or self.complete_percent == 1.0:
                    done = True
        #print(self.vehicle.getCurrentSpeedKmHour())

        self.base.graphicsEngine.renderFrame()

        # Get the frame data as numpy array
        bgr_img = np.frombuffer(self.bgr_tex.getRamImage(), dtype=np.uint8)
        bgr_img.shape = (self.bgr_tex.getYSize(), self.bgr_tex.getXSize(), self.bgr_tex.getNumComponents())

        bgr_img = bgr_img.copy()
        bgr_img = np.flipud(bgr_img[..., [2, 1, 0]])
       # if not os.path.exists("./Testing_shots"):
       #     os.mkdir("./Testing_shots")
        

       # img = Image.fromarray(bgr_img)
        
        
       # if self.num_steps % 10 == 0:
       #     img.save("./Testing_shots/test{}.jpg".format(self.num_steps))
        #print(reward)
        self.num_steps += 1
        if action is None:
            return bgr_img
        else:
            return bgr_img, reward, done, done, {"completeness":self.complete_percent}

    def processInput(self, dt, action):
        engineForce = self.maxEngineForce * action[0]


        brakeForce = self.maxBrakeForce * action[2]

        self.steering += dt * self.steeringIncrement * action[1]
        self.steering = np.clip(self.steering, -self.steeringClamp, self.steeringClamp)

      # Apply steering to front wheels
        self.vehicle.setSteeringValue(self.steering, 0);
        self.vehicle.setSteeringValue(self.steering, 1);


        # 0-1 front wheels 2-3 rare wheels
      # Apply engine and brake to front wheels
        self.vehicle.applyEngineForce(engineForce, 0);
        self.vehicle.applyEngineForce(engineForce, 1);
        self.vehicle.setBrake(brakeForce, 0);
        self.vehicle.setBrake(brakeForce, 1);

    def doExit(self):
        self.cleanup()
        sys.exit(1)

    

    def doReset(self):
        self.cleanup()
        self.setup()

    def toggleWireframe(self):
        self.base.toggleWireframe()

    def toggleTexture(self):
        self.base.toggleTexture()

    def toggleDebug(self):
        if self.debugNP.isHidden():
            self.debugNP.show()
        else:
            self.debugNP.hide()

    def doScreenshot(self):
        self.base.screenshot('Bullet')





if __name__ == "__main__":
    maps = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]
    i = 1
    app = MyEnv(render_mode = "human",view_mode="back-follow", map_file = "./{}/{}.yaml".format(MAPS_DIR, maps[0]), max_n_steps=5)
    
    inputState.watchWithModifiers('forward', 'w')
    #inputState.watchWithModifiers('left', 'a')
    inputState.watchWithModifiers('reverse', 's')
    #inputState.watchWithModifiers('right', 'd')
    inputState.watchWithModifiers('turnLeft', 'a')
    inputState.watchWithModifiers('turnRight', 'd')
    inputState.watchWithModifiers('skip', "z")
    step = 0
    while True:
        action = [0, 0, 0]
        

        if inputState.isSet('forward'):
            action[0] = 1

        if inputState.isSet('reverse'):
            action[2] = 1

        if inputState.isSet('turnLeft'):
            action[1] = 1

        if inputState.isSet('turnRight'):
            action[1] = -1

        obs, reward, done, _, _ = app.step(action)
        print(step ,reward)
        if inputState.isSet("skip"):
            if step > 3:
                done = True
        if done:
            #print(i, maps[i])
            app.reset("{}/{}.yaml".format(MAPS_DIR, maps[i]))
            i += 1
            step = 0
        if i == len(maps):
            i = 0
        step += 1






"""
self.heightImage = PNMImage(257, 257)
        self.heightImage.fillVal(33)
        terrain = GeoMipTerrain("mySimpleTerrain")
        terrain.setHeightfield(self.heightImage)
#terrain.setBruteforce(True)
        terrain.getRoot().setSz(100)
        terrain.getRoot().reparentTo(self.self.base.render)
        terrain.generate()
"""
