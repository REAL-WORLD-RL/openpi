import collections
import dataclasses
import logging
import pathlib

import gymnasium as gym
import imageio
import mani_skill.envs
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from mani_skill.utils.gym_utils import find_max_episode_steps_value 


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # ManiSkill environment-specific parameters
    #################################################################################################################
    env_id: str = "PickCube-v1"  # Task environment ID
    num_envs: int = 1  # Number of parallel environments
    sim_backend: str = "cpu"  # Simulation backend: "cpu" or "gpu"
    render_backend: str = "gpu"  # Rendering backend: "none", "cpu", or "gpu". Use "gpu" for best performance (requires Vulkan)
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_episode_steps: int = 100  # Maximum steps per episode

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/maniskill/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def eval_maniskill(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize ManiSkill environment
    # IMPORTANT: Use the same obs_mode and control_mode as during data collection!
    # Based on the data collection: 
    # - obs_mode="rgb" provides RGB images only (no state in obs, but state available separately)
    # - control_mode="pd_ee_delta_pose" (end-effector delta pose control)
    # - Training data has 7-dim actions: 6D delta pose (3 position + 3 rotation) + 1 gripper

    # Build environment kwargs, conditionally include render_backend
    env_kwargs = {
        "obs_mode": "rgb",
        "control_mode": "pd_ee_delta_pose",
        "render_mode": "rgb_array",
        "sim_backend": args.sim_backend,
        "max_episode_steps": args.max_episode_steps,
        "num_envs": 1,
        "reconfiguration_freq": 1,
    }
    # Only include render_backend if it's not "none"
    if args.render_backend != "none":
        env_kwargs["render_backend"] = args.render_backend

    env = gym.make(args.env_id, **env_kwargs)
    # Test: reset once to inspect observation structure
    test_obs, _ = env.reset(seed=args.seed)
    max_len = find_max_episode_steps_value(env)  # 使用工具函数  

    logging.info(f"Max episode steps: {max_len}")
    logging.info(f"Control mode: {env.control_mode}")
    logging.info(f"Action space: {env.action_space}")
    logging.info(f"Observation keys: {list(test_obs.keys())}")
    
    # Check for images
    if "sensor_data" in test_obs:
        logging.info(f"sensor_data keys: {list(test_obs['sensor_data'].keys())}")
        for sensor_key in test_obs['sensor_data'].keys():
            sensor = test_obs['sensor_data'][sensor_key]
            if isinstance(sensor, dict):
                logging.info(f"  {sensor_key}: {list(sensor.keys())}")
    
    # Check for state
    if "state" in test_obs:
        state_val = test_obs['state']
        if hasattr(state_val, 'shape'):
            logging.info(f"state shape: {state_val.shape}")
        elif hasattr(state_val, '__len__'):
            logging.info(f"state length: {len(state_val)}")
    
    if "agent" in test_obs:
        logging.info(f"agent keys: {list(test_obs['agent'].keys())}")
        total_agent_dims = 0
        for k, v in test_obs['agent'].items():
            if hasattr(v, 'shape'):
                dim_size = v.numel() if hasattr(v, 'numel') else (v.size if not callable(v.size) else np.prod(v.shape))
                logging.info(f"  {k}: {v.shape}, elements: {dim_size}")
                total_agent_dims += dim_size
        logging.info(f"Total agent dimensions: {total_agent_dims}")
    
    # Check if extra contains object/goal info that contributes to the 42-dim state
    if "extra" in test_obs:
        if isinstance(test_obs['extra'], dict):
            logging.info(f"extra keys: {list(test_obs['extra'].keys())}")
            extra_total_dims = 0
            for k, v in test_obs['extra'].items():
                if hasattr(v, 'shape'):
                    dim_size = v.numel() if hasattr(v, 'numel') else np.prod(v.shape)
                    logging.info(f"  {k}: {v.shape}, elements: {dim_size}")
                    extra_total_dims += dim_size
            logging.info(f"Total extra dimensions: {extra_total_dims}")
            logging.info(f"Agent + Extra = {total_agent_dims} + {extra_total_dims} = {total_agent_dims + extra_total_dims}")
    
    # Get task description from environment
    task_description = _get_task_description(args.env_id)
    logging.info(f"Task description: {task_description}")

    # Initialize policy client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        logging.info(f"\nStarting episode {episode_idx + 1}/{args.num_trials_per_task}...")
        
        # Reset environment
        obs, info = env.reset(seed=args.seed + episode_idx)
        action_plan = collections.deque()
        
        # Setup
        t = 0
        replay_images = []  # Robot sensor camera (model input)
        replay_render_images = []  # Human render camera (for visualization)
        done = False
        success = False

        while t < max_len:
            #logging.info(f"t: {t}")
            try:
                # Get RGB image from base_camera
                # obs_mode="rgb+state" provides:
                # - sensor_data["base_camera"]["rgb"] for images
                # - state for flattened state vector
                if "sensor_data" in obs and "base_camera" in obs["sensor_data"]:
                    rgb_img = obs["sensor_data"]["base_camera"]["rgb"]
                elif "image" in obs:
                    rgb_img = obs["image"]
                else:
                    raise ValueError(f"Cannot find RGB image in observation. Keys: {list(obs.keys())}")
                
                # Convert torch tensor to numpy if needed
                if hasattr(rgb_img, 'cpu'):
                    rgb_img = rgb_img.cpu().numpy()
                
                # Handle batch dimension if present: (1, H, W, 3) -> (H, W, 3)
                if rgb_img.ndim == 4:
                    rgb_img = rgb_img[0]
                
                # Convert to uint8 and resize
                img = image_tools.convert_to_uint8(rgb_img)
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                
                # Check for hand_camera (optional second camera)
                hand_img = None
                if "sensor_data" in obs and "hand_camera" in obs["sensor_data"]:
                    hand_rgb_img = obs["sensor_data"]["hand_camera"]["rgb"]
                    
                    # Convert torch tensor to numpy if needed
                    if hasattr(hand_rgb_img, 'cpu'):
                        hand_rgb_img = hand_rgb_img.cpu().numpy()
                    
                    # Handle batch dimension if present: (1, H, W, 3) -> (H, W, 3)
                    if hand_rgb_img.ndim == 4:
                        hand_rgb_img = hand_rgb_img[0]
                    
                    # Convert to uint8 and resize
                    hand_img = image_tools.convert_to_uint8(hand_rgb_img)
                    hand_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(hand_img, args.resize_size, args.resize_size)
                    )
                    
                    if t == 0 and episode_idx == 0:
                        logging.info(f"Hand camera detected: {hand_img.shape}")
                
                # Get proprioceptive state
                # State format: 9-dim = 7 joint positions + 2 gripper positions
                # - First 7 dims: joint positions (qpos[0:7])
                # - Last 2 dims: gripper positions (qpos[7:9])
                if "agent" in obs:
                    # Extract qpos from agent observation
                    if "qpos" in obs["agent"]:
                        qpos = obs["agent"]["qpos"]
                        if hasattr(qpos, 'cpu'):
                            qpos = qpos.cpu().numpy()
                        qpos = qpos.flatten()
                        
                        # State = first 7 joints + 2 gripper positions = 9 dims
                        # qpos should be 9-dim: [7 joints + 2 gripper]
                        if len(qpos) >= 9:
                            state = qpos[:9]  # Take first 9 dims (7 joints + 2 gripper)
                        elif len(qpos) == 7:
                            # If only 7 joints, need to get gripper separately
                            if "gripper_qpos" in obs["agent"]:
                                gripper_qpos = obs["agent"]["gripper_qpos"]
                                if hasattr(gripper_qpos, 'cpu'):
                                    gripper_qpos = gripper_qpos.cpu().numpy()
                                gripper_qpos = gripper_qpos.flatten()
                                state = np.concatenate([qpos[:7], gripper_qpos[:2]])
                            else:
                                raise ValueError(f"qpos has {len(qpos)} dims but no gripper_qpos found")
                        else:
                            raise ValueError(f"Unexpected qpos length: {len(qpos)}, expected 7 or 9")
                    else:
                        raise ValueError("'qpos' not found in obs['agent']")
                    
                    # Output state dimensions
                    if t == 0 and episode_idx == 0:
                        logging.info(f"State dimension: {state.shape[0]} (7 joints + 2 gripper)")
                        logging.info(f"State format: joints[0:7]={state[:7]}, gripper[7:9]={state[7:9]}")
                elif "state" in obs:
                    # Fallback: if state is directly available
                    state = obs["state"]
                    if hasattr(state, 'cpu'):
                        state = state.cpu().numpy()
                    state = state.flatten()
                    if t == 0 and episode_idx == 0:
                        logging.info(f"State dimension: {state.shape[0]}")
                else:
                    raise ValueError(f"Neither 'agent' nor 'state' key found in observation! Keys: {list(obs.keys())}")


                # Save preprocessed image for replay video (robot sensor view)
                replay_images.append(img)
                
                # Also get human render camera view for visualization
                try:
                    render_img = env.render()  # Get render camera image
                    if render_img is not None:
                        # Convert to numpy if needed
                        if hasattr(render_img, 'cpu'):
                            render_img = render_img.cpu().numpy()
                        # Remove batch dimension if present
                        if render_img.ndim == 4:
                            render_img = render_img[0]
                        # Convert and resize
                        render_img = image_tools.convert_to_uint8(render_img)
                        render_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(render_img, args.resize_size, args.resize_size)
                        )
                        replay_render_images.append(render_img)
                except Exception as e:
                    if t == 0 and episode_idx == 0:
                        logging.warning(f"Could not get render camera: {e}")

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Ensure state is 1D array
                    if state.ndim > 1:
                        state = state.flatten()
                    
                    # Debug: check shapes before sending (only print once)
                    if t == 0 and episode_idx == 0:
                        logging.info(f"Debug shapes - img: {img.shape}, state: {state.shape}")
                        if hand_img is not None:
                            logging.info(f"Debug shapes - hand_img: {hand_img.shape}")
                    
                    # Prepare observations dict matching maniskill5_policy format
                    element = {
                        "observation.images.base_camera": img,
                        "observation.state": state,
                        "task": str(task_description),
                    }
                    
                    # Add hand camera if available (for dual-camera datasets)
                    if hand_img is not None:
                        element["observation.images.hand_camera"] = hand_img

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    # action_chunk = [np.random.uniform(
                    #     low=env.action_space.low,
                    #     high=env.action_space.high,
                    #     size=env.action_space.shape
                    # )]*10
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                # Execute action in environment
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                
                # Handle batch dimension first
                if action.ndim > 1:
                    action = action.squeeze(0)
                
                # IMPORTANT: Handle different control modes dynamically
                # - pd_ee_delta_pose: expects 7-dim actions (6D delta pose + 1 gripper)
                # - pd_joint_pos: expects 8-dim actions (7 joints + 1 gripper)
                expected_action_dim = env.action_space.shape[0]
                
                if action.shape[-1] > expected_action_dim:
                    if t == 0 and episode_idx == 0:
                        logging.info(f"Model outputs {action.shape[-1]}-dim actions, using first {expected_action_dim} dims for {env.control_mode}")
                    action = action[:expected_action_dim]
                elif action.shape[-1] != expected_action_dim:
                    raise ValueError(f"Expected {expected_action_dim}-dim action for {env.control_mode}, got {action.shape[-1]}")
                
                '''# TEMPORARY: Replace with random action for testing
                action = np.random.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.shape
                )
                if t == 0 and episode_idx == 0:
                    logging.info(f"🎲 USING RANDOM ACTIONS for testing!")
                    logging.info(f"   Action space: {env.action_space}")
                    logging.info(f"   Random action sample: {action}")'''
                    
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Check for success
                if done:
                    # ManiSkill typically stores success in info
                    success = info.get("success", False)
                    if success:
                        total_successes += 1
                    break
                if truncated:
                    logging.info("truncated, t:", t)
                
                t += 1

            except KeyboardInterrupt:
                logging.info("Interrupted by user")
                break
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback
                traceback.print_exc()
                break

        total_episodes += 1

        # Save replay videos
        suffix = "success" if success else "failure"
        task_segment = task_description.replace(" ", "_")
        
        # Save robot sensor camera video (what the robot sees - model input)
        if replay_images:
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}_sensor.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logging.info(f"Saving SENSOR video ({len(replay_images)} frames): {video_path}")
                imageio.mimwrite(video_path, replay_images, fps=10)
                logging.info(f"✓ Sensor video saved")
            except Exception as e:
                logging.error(f"✗ Failed to save sensor video: {e}")
        else:
            logging.warning(f"No sensor images for episode {episode_idx + 1}")
        
        # Save human render camera video (visualization view - for comparison)
        if replay_render_images:
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:03d}_{suffix}_render.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                logging.info(f"Saving RENDER video ({len(replay_render_images)} frames): {video_path}")
                imageio.mimwrite(video_path, replay_render_images, fps=10)
                logging.info(f"✓ Render video saved")
            except Exception as e:
                logging.error(f"✗ Failed to save render video: {e}")
        else:
            if episode_idx == 0:
                logging.warning(f"No render images available (render camera may not be configured)")

        # Log current results
        logging.info(f"Episode {episode_idx + 1}: Success={success}, Steps={t}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    # Log final results
    logging.info(f"\n{'='*60}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes) * 100:.1f}%")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    logging.info(f"{'='*60}")

    env.close()


def _get_task_description(env_id: str) -> str:
    """Get human-readable task description for the environment."""
    # Map common ManiSkill task IDs to descriptions
    task_descriptions = {
        "PickCube-v1": "pick up the red cube and move it to the goal",
        "StackCube-v1": "stack the red cube on the green cube",
        "PegInsertionSide-v1": "insert the peg into the hole from the side",
        "PlugCharger-v1": "plug the charger into the socket",
        "PushChair-v1": "push the chair to the target location",
        "OpenCabinetDoor-v1": "open the cabinet door",
        "OpenCabinetDrawer-v1": "open the cabinet drawer",
        "PushCube-v1": "push the cube to the target location",
        "TurnFaucet-v1": "turn on the faucet",
    }
    
    return task_descriptions.get(env_id, env_id.replace("-", " ").replace("v1", "").strip())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_maniskill)

