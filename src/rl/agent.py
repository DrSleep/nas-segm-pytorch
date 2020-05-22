"""Reinforcement Learning-based agent"""

from .gradient_estimators import REINFORCE, PPO


def create_agent(
    enc_num_layers,
    num_ops,
    num_agg_ops,
    lstm_hidden_size,
    lstm_num_layers,
    dec_num_cells,
    cell_num_layers,
    cell_max_repeat,
    cell_max_stride,
    ctrl_lr,
    ctrl_baseline_decay,
    ctrl_agent,
    ctrl_version='cvpr',
):
    """Create Agent

    Args:
      enc_num_layers (int) : size of initial sampling pool, number of encoder outputs
      num_ops (int) : number of unique operations
      num_agg_ops (int) : number of unique aggregation operations
      lstm_hidden_size (int) : number of neurons in RNN's hidden layer
      lstm_num_layers (int) : number of LSTM layers
      dec_num_cells (int) : number of cells in the decoder
      cell_num_layers (int) : number of layers in a cell
      cell_max_repeat (int) : maximum number of repeats the cell (template) can be repeated.
                              only valid for the 'wacv' controller
      cell_max_stride (int) : max stride of the cell (template). only for 'wacv'
      ctrl_lr (float) : controller's learning rate
      ctrl_baseline_decay (float) : controller's baseline's decay
      ctrl_agent (str) : type of agent's controller
      ctrl_version (str, either 'cvpr' or 'wacv') : type of microcontroller

    Returns:
      controller net that provides the sample() method
      gradient estimator

    """
    if ctrl_version == 'cvpr':
        from rl.micro_controllers import MicroController as Controller
    elif ctrl_version == 'wacv':
        from rl.micro_controllers import TemplateController as Controller

    controller = Controller(
        enc_num_layers=enc_num_layers,
        num_ops=num_ops,
        num_agg_ops=num_agg_ops,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dec_num_cells=dec_num_cells,
        cell_num_layers=cell_num_layers,
        cell_max_repeat=cell_max_repeat,
        cell_max_stride=cell_max_stride,
    )
    if ctrl_agent == "ppo":
        agent = PPO(
            controller,
            clip_param=0.1,
            lr=ctrl_lr,
            baseline_decay=ctrl_baseline_decay,
            action_size=controller.action_size(),
        )
    elif ctrl_agent == "reinforce":
        agent = REINFORCE(controller, lr=ctrl_lr, baseline_decay=ctrl_baseline_decay)
    return agent


def train_agent(agent, sample):
    """Training controller"""
    config, reward, entropy, log_prob = sample
    action = agent.controller.config2action(config)
    loss, dist_entropy = agent.update((reward, action, log_prob))
