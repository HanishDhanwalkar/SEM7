import gymnasium
import numpy as np


go1_xml_file = 'unitree_go1/scene.xml'


env = gymnasium.make(
    'Ant-v5',
    xml_file=go1_xml_file,
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,  # kept the game as the 'Ant' environment
    exclude_current_positions_from_observation=False,  # kept the game as the 'Ant' environment
    reset_noise_scale=0.1,
    frame_skip=25,
    # max_episode_steps=10000,
    render_mode='human'
)


observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    
    # print(f"Observation: {observation}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
    
    episode_over = terminated or truncated
    
    

env.close()