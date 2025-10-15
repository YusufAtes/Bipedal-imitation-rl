# import the agent and its default configuration
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_biped import BipedEnv


from skrl.memories.torch import RandomMemory

# main training memory (on-policy; no need to be huge)
memory = RandomMemory(memory_size=2048, num_envs=1, device=env.device)

# short FIFO buffer of AMP observations from the policy (for discriminator)
reply_buffer = RandomMemory(memory_size=50000, num_envs=1, device=env.device)


# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training
models["discriminator"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = AMP_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
# (assuming defined memories for motion <motion_dataset> and <reply_buffer>)
# (assuming defined methods to collect motion <collect_reference_motions> and <collect_observation>)
agent = AMP(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            amp_observation_space=env.amp_observation_space,
            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            collect_reference_motions=collect_reference_motions,
            collect_observation=collect_observation)