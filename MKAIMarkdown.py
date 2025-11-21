# %% [markdown]
# # THIS CODE IS BUILT TO RUN ON GOOGLE COLAB ONLY
# 
# Setup Steps (Pre-Run):
# 
# 1) Click on the Files icon to the right on the bottom. By default, you should be in the /content folder, but if not, then navigate to /content from the root (right-click on /content and choose "Open").
# 
# 2) Upload the ROM included to the /content folder.
# 
# 3) It is recommended to "Disconnect and delete runtime" (under Runtime menu) before running just to ensure that old data from previous runtimes will not be included.
# 
# On first run, the /video folder will be created under /content, and a video of the frames will be placed there with a .json file.  These will be replaced each time the program is run, so save a copy of the video after each run (the video must be recorded each time because Colab does not have a direct display capability).  The .json file does not serve a purpose for this project.

# %%
# Installs and Setup
!apt-get update
!apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev

!pip install opencv-python
!pip install optuna
!pip install stable-baselines3[extra]
!pip install stable-retro gymnasium
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

LOG_DIR = './logs/'   # Logging directory
OPT_DIR = './opt/'    # Optimized model directory
CHK_DIR = './train/'  # Training directory

# %%
# Imports

import cv2
import gymnasium as gym
import numpy as np
import optuna
import os
import pyglet
import retro
import retro.data
import subprocess
import time
import warnings

from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary
from gymnasium.wrappers import RecordVideo

from IPython import get_ipython

from matplotlib import pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# %%
# Load the Rom - retro.data.list_games() # to see supported

!python -m retro.import .

# %%
# Objective Function

def objective(trial):

    return {

        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),

    }

# %%
# Optimize Function

def optimize(trial):

    try:

        model_params = objective(trial)

        env = MortalKombat()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        model = PPO('CnnPolicy', env, verbose = 0, tensorboard_log = LOG_DIR, **model_params)
        model.learn(total_timesteps = 100) #big 0's = bigger trial

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 1)
        env.close()

        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:

        return -1


# %%
# Main class

class MortalKombat(gym.Env):

    def __init__(self, render_mode = 'rgb_array'):

        super().__init__()
        self.observation_space = Box(low = 0, high = 255, shape = (84, 84, 1), dtype = np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game = 'MortalKombat-Genesis', use_restricted_actions = retro.Actions.FILTERED, render_mode='rgb_array')
        self.previous_frame = None
        self.score = 0
        self.frame_counter = 0
        self.game_over = False
        self.game_start = False

    def step(self, action):

        obs, reward, terminated, _, info = self.game.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs
        reward = info['score'] - self.score
        self.score = info['score']
        truncated = False
        done = terminated or self.game_over
        info = info
        self.frame_counter += 1
        return frame_delta, reward, terminated, truncated, info

    def render(self, *args, **kwargs):

        self.game.render()

    def reset(self, **kwargs):

        _ = kwargs.get("seed", None)
        obs, info = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs, info

    def close(self):

        self.game.close()

    def preprocess(self, observation):

        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation = cv2.INTER_CUBIC)
        channels  = np.reshape(resize, (84, 84, 1))
        return channels


# %%
# Callback Function

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose = 1):

        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)

    def _on_step(self):

        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

# %%
# Main TRAINING Block

try:

    env.close()

except NameError:

    pass

# Create study based on optimize function
# This is to create the trial files that will prime the later run
study = optuna.create_study(direction = 'maximize')
study.optimize(optimize, n_trials = 5, n_jobs = 1)
study.best_params
study.best_trial.number

# Set up callback - check_freq = count at which every data is recorded
callback = TrainAndLoggingCallback(check_freq = 10000, save_path = CHK_DIR)

# Train model
env = MortalKombat()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# Take the best params from the study
model_params = study.best_params

# Apply the PPO policy to the model using the best params
model = PPO('CnnPolicy', env, verbose = 1, tensorboard_log = LOG_DIR, **model_params)

# %%
# the best trial model - hyperparameter tuned
model.load(os.path.join(OPT_DIR, 'trial_{}_best_model'.format(study.best_trial.number)))

# %%
# Now we let the model learn
# Increase total_timesteps by order of 10k to generate more tests to learn from
model.learn(total_timesteps = 10000, callback = callback) #extra 0's = bigger trial iterations

# %%
# Take the best model created from the output
model = PPO.load('./train/best_model_10000.zip')

# %%
# Evaluate the mean reward
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 5, render = False)

# %%
# Check the mean reward here - is it > 0 ?
mean_reward

# %%
# With Training
try:

    env.close()

except NameError:

    pass

env = MortalKombat()
video_every = 100000
env.render_mode = 'rgb_array'
env = RecordVideo(env, "./video", episode_trigger=lambda episode_id: (episode_id % video_every) == 0)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

obs = env.reset()
done = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.venv.envs[0].render()
        action, _ = model.predict(obs[0], deterministic=True)
        action = np.array(env.action_space.sample()).astype(int)
        obs, reward, terminated, info = env.step([action])
        truncated = info[0].get('TimeLimit.truncated', False)
        done = terminated or truncated
        if reward > 0:
            print(reward)
env.close()
info

# %%
# Without Training

try:

    env.close()

except NameError:

    pass

env = retro.make(game='MortalKombat-Genesis', render_mode='rgb_array')
video_every = 100000
env = RecordVideo(env, "./video", episode_trigger=lambda episode_id: (episode_id % video_every) == 0)

obs = env.reset()
done = False
for game in range(1):
    while not done:
        if done:
            obs = env.reset()
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        if reward > 0:
            print(reward)
env.close()
info


