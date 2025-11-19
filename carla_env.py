# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
from matplotlib import cm
from skimage.transform import resize
import pygame

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *  # assumes helper functions (display_to_rgb, rgb_to_display_surface, get_pos, get_lane_dis, ...)

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator.
     - 4 RGB cameras (front, front-right, front-left, rear-ish)
     - 1 LiDAR
     - 1 Radar
     - 1 Collision sensor
  """

  def __init__(self, params):
    # === parameters ===
    self.display_size = params['display_size']
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range / self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params.get('display_route', False)

    # action space
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']]
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                               params['continuous_steer_range'][0]]),
                                     np.array([params['continuous_accel_range'][1],
                                               params['continuous_steer_range'][1]]),
                                     dtype=np.float32)

    # observation: keep the original (5, obs_size, obs_size, 3) shape:
    self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(5, self.obs_size, self.obs_size, 3),
                                        dtype=np.float32)

    # connect to CARLA
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(params.get('client_timeout', 10.0))
    self.world = client.load_world(params['town'])
    print('Carla server connected!')

    # try set weather (non-fatal)
    try:
      self.world.set_weather(carla.WeatherParameters.ClearNoon)
    except Exception:
      pass

    # spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if loc is not None:
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # blueprints
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # collision blueprint
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # lidar blueprint
    self.lidar_height = 1.8
    self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '64')
    self.lidar_bp.set_attribute('range', '100.0')
    self.lidar_bp.set_attribute('upper_fov', '15')
    self.lidar_bp.set_attribute('lower_fov', '-25')
    self.lidar_bp.set_attribute('rotation_frequency', str(1.0 / max(1e-6, self.dt)))
    self.lidar_bp.set_attribute('points_per_second', '500000')

    # radar blueprint
    self.radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
    self.radar_bp.set_attribute('horizontal_fov', '30.0')
    self.radar_bp.set_attribute('vertical_fov', '30.0')
    self.radar_bp.set_attribute('points_per_second', '10000')
    self.radar_trans = carla.Transform(carla.Location(z=2))

    # camera blueprint and buffer
    self.camera_img = np.zeros((4, self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    self.camera_trans = carla.Transform(carla.Location(x=1.5, z=1.5))
    self.camera_trans2 = carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0))
    self.camera_trans3 = carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))
    self.camera_trans4 = carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0))

    # simulation settings
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # bookkeeping
    self.reset_step = 0
    self.total_step = 0
    self.collision_hist = []
    self.collision_hist_l = 1

    # actor handles
    self.ego = None
    self.collision_sensor = None
    self.lidar_sensor = None
    self.radar_sensor = None
    self.camera_sensor = None
    self.camera_sensor2 = None
    self.camera_sensor3 = None
    self.camera_sensor4 = None

    # lidar image buffer (always numeric dtype)
    self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    # prepare renderer
    self._init_renderer()

    # routeplanner placeholder (set after ego spawn)
    self.routeplanner = None

  # -----------------------
  # CALLBACKS (class-level)
  # -----------------------

  def _on_collision(self, event):
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    self.collision_hist.append(intensity)
    if len(self.collision_hist) > self.collision_hist_l:
      self.collision_hist.pop(0)

  def _radar_callback(self, data):
    # convert radar to points/colors for potential use; keep robust
    if len(data) == 0:
      self._radar_points = np.zeros((0,3), dtype=np.float32)
      self._radar_colors = np.zeros((0,3), dtype=np.float32)
      return

    radar_data = np.zeros((len(data), 4), dtype=np.float32)
    for i, detection in enumerate(data):
      x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
      y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
      z = detection.depth * math.sin(detection.altitude)
      radar_data[i, :] = [x, y, z, detection.velocity]

    intensity = np.abs(radar_data[:, -1])
    intensity = np.clip(intensity, 1e-6, None)
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
    COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))[:, :3]
    int_color = np.c_[
      np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
      np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
      np.interp(intensity_col, COOL_RANGE, COOL[:, 2]),
    ]
    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    self._radar_points = points
    self._radar_colors = int_color

  def _lidar_callback(self, point_cloud):
    # Robustly convert raw lidar into a BEV image (numeric array)
    try:
      data = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
      if data.size == 0:
        # keep existing lidar_img but ensure dtype
        self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        return
      points = np.reshape(data, (-1, 4))  # x,y,z,intensity
    except Exception:
      self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
      return

    xs = points[:, 0]
    ys = points[:, 1]
    intens = points[:, 3]
    # clip intensities and avoid log(0) problems if later used
    intens = np.clip(intens, 0.0, 1.0)

    half_range = self.obs_range / 2.0
    mask = (xs > -half_range) & (xs < half_range) & (ys > -half_range) & (ys < half_range)
    if not np.any(mask):
      self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
      return

    xs = xs[mask]
    ys = ys[mask]
    ic = intens[mask]

    # map coordinates to pixel indices
    px = ((xs + half_range) / (2.0 * half_range) * (self.obs_size - 1)).astype(np.int32)
    py = ((ys + half_range) / (2.0 * half_range) * (self.obs_size - 1)).astype(np.int32)

    # clamp indices
    px = np.clip(px, 0, self.obs_size - 1)
    py = np.clip(py, 0, self.obs_size - 1)

    # make image and color by plasma colormap
    cmap = (VIRIDIS * 255).astype(np.uint8)
    idx = np.clip((ic * (cmap.shape[0] - 1)).astype(np.int32), 0, cmap.shape[0] - 1)

    img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    img[py, px] = cmap[idx, :3]
    self.lidar_img = img

  # camera callbacks (class-level)
  def _camera_callback_0(self, image):
    try:
      arr = np.frombuffer(image.raw_data, dtype=np.uint8)
      arr = arr.reshape((image.height, image.width, 4))
      arr = arr[:, :, :3]
      arr = arr[:, :, ::-1]  # BGR->RGB
      self.camera_img[0] = arr
    except Exception:
      pass

  def _camera_callback_1(self, image):
    try:
      arr = np.frombuffer(image.raw_data, dtype=np.uint8)
      arr = arr.reshape((image.height, image.width, 4))
      arr = arr[:, :, :3]
      arr = arr[:, :, ::-1]
      self.camera_img[1] = arr
    except Exception:
      pass

  def _camera_callback_2(self, image):
    try:
      arr = np.frombuffer(image.raw_data, dtype=np.uint8)
      arr = arr.reshape((image.height, image.width, 4))
      arr = arr[:, :, :3]
      arr = arr[:, :, ::-1]
      self.camera_img[2] = arr
    except Exception:
      pass

  def _camera_callback_3(self, image):
    try:
      arr = np.frombuffer(image.raw_data, dtype=np.uint8)
      arr = arr.reshape((image.height, image.width, 4))
      arr = arr[:, :, :3]
      arr = arr[:, :, ::-1]
      self.camera_img[3] = arr
    except Exception:
      pass

  # -----------------------
  # SENSOR MANAGEMENT
  # -----------------------
  def _cleanup_sensors(self):
    """Stop and destroy any sensors that were spawned."""
    sensors = [
      self.camera_sensor, self.camera_sensor2, self.camera_sensor3, self.camera_sensor4,
      self.lidar_sensor, self.radar_sensor, self.collision_sensor
    ]
    for s in sensors:
      try:
        if s is not None:
          try:
            s.stop()
          except Exception:
            pass
          s.destroy()
      except Exception:
        pass

    # clear references
    self.camera_sensor = None
    self.camera_sensor2 = None
    self.camera_sensor3 = None
    self.camera_sensor4 = None
    self.lidar_sensor = None
    self.radar_sensor = None
    self.collision_sensor = None

  # -----------------------
  # GYM API
  # -----------------------
  def reset(self):
    # cleanup any leftover sensors/actors
    self._cleanup_sensors()
    # remove actors of interest
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb',
                            'vehicle.*', 'controller.ai.walker', 'walker.*'])
    # disable sync to spawn safely
    self._set_synchronous_mode(False)

    # spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # actor polygons
    self.vehicle_polygons = []
    self.vehicle_polygons.append(self._get_actor_polygons('vehicle.*'))
    self.walker_polygons = []
    self.walker_polygons.append(self._get_actor_polygons('walker.*'))

    # spawn ego vehicle (try multiple times)
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        raise RuntimeError("Failed to spawn ego vehicle after {} attempts".format(self.max_ego_spawn_times))
      transform = random.choice(self.vehicle_spawn_points)
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_sensor.listen(lambda event: self._on_collision(event))
    self.collision_hist = []

    # radar
    self.radar_sensor = self.world.spawn_actor(self.radar_bp, self.radar_trans, attach_to=self.ego)
    self.radar_sensor.listen(lambda data: self._radar_callback(data))

    # lidar
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    self.lidar_sensor.listen(lambda data: self._lidar_callback(data))

    # cameras
    self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
    self.camera_sensor.listen(lambda data: self._camera_callback_0(data))

    self.camera_sensor2 = self.world.spawn_actor(self.camera_bp, self.camera_trans2, attach_to=self.ego)
    self.camera_sensor2.listen(lambda data: self._camera_callback_1(data))

    self.camera_sensor3 = self.world.spawn_actor(self.camera_bp, self.camera_trans3, attach_to=self.ego)
    self.camera_sensor3.listen(lambda data: self._camera_callback_2(data))

    self.camera_sensor4 = self.world.spawn_actor(self.camera_bp, self.camera_trans4, attach_to=self.ego)
    self.camera_sensor4.listen(lambda data: self._camera_callback_3(data))

    # reset counters
    self.time_step = 0
    self.reset_step += 1

    # enable synchronous mode after spawning actors
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # route planner and rendering setup
    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
    self.birdeye_render.set_hero(self.ego, self.ego.id)

    # ensure lidar_img exists and is numeric
    if not isinstance(self.lidar_img, np.ndarray) or self.lidar_img.dtype == np.object:
      self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    return self._get_obs()

  def step(self, action):
    # compute accel / steer
    if self.discrete:
      acc = self.discrete_act[0][action // self.n_steer]
      steer = self.discrete_act[1][action % self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    if acc > 0:
      throttle = np.clip(acc / 3, 0, 1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc / 8, 0, 1)

    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    # tick world (synchronous mode expected)
    try:
      self.world.tick()
    except Exception:
      # fallback, try apply settings then tick
      try:
        self.world.apply_settings(self.settings)
        self.world.tick()
      except Exception:
        pass

    # bookkeeping
    now = datetime.now()
    if not hasattr(self, 'dt0'):
      self.dt0 = now
    process_time = now - self.dt0
    if process_time.total_seconds() > 0:
      sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
      sys.stdout.flush()
    self.dt0 = now

    # update actor polygons buffer
    self.vehicle_polygons.append(self._get_actor_polygons('vehicle.*'))
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    self.walker_polygons.append(self._get_actor_polygons('walker.*'))
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route plan step
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    info = {'waypoints': self.waypoints, 'vehicle_front': self.vehicle_front}

    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode='human'):
    # we already blit to pygame in _get_obs; nothing special to do here
    pass

  # -----------------------
  # HELPERS
  # -----------------------
  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    pygame.init()
    self.display = pygame.display.set_mode(
      (self.display_size * 6, self.display_size),
      pygame.HWSURFACE | pygame.DOUBLEBUF)
    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous=True):
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot(enabled=True, tm_port=4050)
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)
    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      walker_controller_actor.start()
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      walker_controller_actor.set_max_speed(1 + random.random())
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    vehicle = None
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break
    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
    if vehicle is not None:
      self.ego = vehicle
      return True
    return False

  def _get_actor_polygons(self, filt):
    actor_poly_dict = {}
    for actor in self.world.get_actors().filter(filt):
      try:
        trans = actor.get_transform()
        x = trans.location.x
        y = trans.location.y
        yaw = trans.rotation.yaw / 180 * np.pi
        bb = actor.bounding_box
        l = bb.extent.x
        w = bb.extent.y
        poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
        actor_poly_dict[actor.id] = poly
      except Exception:
        pass
    return actor_poly_dict

  def _get_obs(self):
    # birdeye render
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.obs_size)

    # blit birdeye
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    # lidar image (already numeric)
    lidar_arr = np.zeros((1, self.obs_size, self.obs_size, 3), dtype=np.float32)
    try:
      # ensure lidar_img is numeric ndarray
      if not isinstance(self.lidar_img, np.ndarray):
        self.lidar_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
      lidar_resized = resize(self.lidar_img, (self.obs_size, self.obs_size, 3), preserve_range=True)
      lidar_arr[0] = (lidar_resized).astype(np.float32)
    except Exception:
      # fallback blank lidar
      lidar_arr[0] = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.float32)

    lidar_surface = rgb_to_display_surface(lidar_arr[0], self.display_size)
    self.display.blit(lidar_surface, (self.display_size * 1, 0))

    # camera images: ensure numeric and resized
    camera = (resize(self.camera_img, (4, self.obs_size, self.obs_size, 3), preserve_range=True)).astype(np.float32)
    # blit cameras in the same layout as your original code
    camera_surface = rgb_to_display_surface(camera[0], self.display_size)
    self.display.blit(camera_surface, (self.display_size * 3, 0))

    camera_surface2 = rgb_to_display_surface(camera[1], self.display_size)
    self.display.blit(camera_surface2, (self.display_size * 2, 0))

    camera_surface3 = rgb_to_display_surface(camera[2], self.display_size)
    self.display.blit(camera_surface3, (self.display_size * 4, 0))

    camera_surface4 = rgb_to_display_surface(camera[3], self.display_size)
    self.display.blit(camera_surface4, (self.display_size * 5, 0))

    pygame.display.flip()

    obs = {
      'camera': camera,
      'lidar': lidar_arr,
      'birdeye': birdeye.astype(np.uint8)
    }

    # return concatenated: [4 cameras, lidar]
    return np.concatenate((obs['camera'], obs['lidar']))

  def _get_reward(self):
    v = self.ego.get_velocity()
    speed = math.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)

    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    r_steer = -self.ego.get_control().steer**2

    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1

    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = 200 * r_collision + 5 * r_speed + 10 * r_fast + 2 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1
    return r

  def _terminal(self):
    ego_x, ego_y = get_pos(self.ego)

    if len(self.collision_hist) > 0:
      return True
    if self.time_step > self.max_time_episode:
      return True
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True
    return False

  def _clear_all_actors(self, actor_filters):
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        try:
          if actor.is_alive:
            if actor.type_id == 'controller.ai.walker':
              actor.stop()
            actor.destroy()
        except Exception:
          pass

  def close(self):
    try:
      self._cleanup_sensors()
    except Exception:
      pass
    try:
      self.settings.synchronous_mode = False
      self.world.apply_settings(self.settings)
    except Exception:
      pass


