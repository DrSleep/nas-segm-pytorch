"""RL Controller"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.utils import calc_prob, compute_critic_logits, sample_logits, torch_long


class MicroController(nn.Module):
    """ Stack LSTM controller based on ENAS

    https://arxiv.org/abs/1802.03268

    With modification that indices and ops chosen with linear classifier
    Samples decoder structure and connections

    """

    def __init__(
        self,
        enc_num_layers,
        num_ops,
        lstm_hidden_size=100,
        lstm_num_layers=2,
        dec_num_cells=3,
        cell_num_layers=4,
        **kwargs,
    ):
        """
        Args:
          enc_num_layers (int): encoder input scales
          num_ops (int): number of operations
          lstm_hidden_size (int): number of hidden units of LSTM
          lstm_num_layers (int): number of LSTM layers
          dec_num_cells (int): number of cells in the decoder
          cell_num_layers (int): numebr of layers in each cell

        """
        super(MicroController, self).__init__()

        # Each decoder block consists of 2 cells, independently applied to 2 inputs
        self._num_cells_per_decoder_block = 2
        # Each cell takes 1 input
        self._num_inputs_per_cell = 1
        # Applies a single operation per input
        self._num_ops_per_cell_input = 1
        # Produces 1 output
        self._num_outputs_per_cell = 1
        # Within the cell, each layer (except the first one) takes 2 inputs and for each applies an operation
        self._num_inputs_per_cell_layer = 2
        self._num_ops_per_cell_layer_input = 1
        # Each cell layer produces 1 output
        self._num_outputs_per_cell_layer = 1

        # Customisable configuration
        self.cell_num_layers = cell_num_layers
        self.dec_num_cells = dec_num_cells
        self.enc_num_layers = enc_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # Number of decoder connections
        total_num_cell_inputs = (
            self._num_cells_per_decoder_block
            * self._num_inputs_per_cell
            * self.dec_num_cells
        )
        # Number of connections within the cell
        total_num_cell_connections = self._num_inputs_per_cell_layer * (
            self.cell_num_layers - 1
        )  # -1 for the input layer
        # Number of ops within the cell
        total_num_cell_ops = (
            # Within the cell we do not need to predict the first input,
            # but we still keep it around as dummy,
            # hence need to multiply by two
            2 * self._num_inputs_per_cell * self._num_ops_per_cell_input
            + total_num_cell_connections * self._num_ops_per_cell_layer_input
        )
        self._action_len = (
            total_num_cell_inputs + total_num_cell_connections + total_num_cell_ops
        )

        # Controller
        self.rnn = nn.LSTM(lstm_hidden_size, lstm_hidden_size, lstm_num_layers)
        self.enc_op = nn.Embedding(num_ops, lstm_hidden_size)
        self.linear_op = nn.Linear(lstm_hidden_size, num_ops)
        self.g_emb = nn.Parameter(torch.zeros(1, 1, lstm_hidden_size))

        # Connection predictions: predicting inputs to the cell
        conn_fcs = []
        for i in range(self.dec_num_cells):
            for _ in range(
                self._num_cells_per_decoder_block * self._num_inputs_per_cell
            ):
                # each time the sampling pool grows by number of outputs in the cell
                conn_fcs.append(
                    nn.Linear(
                        lstm_hidden_size,
                        self.enc_num_layers + i * self._num_outputs_per_cell,
                    )
                )
        self.conn_fcs = nn.ModuleList(conn_fcs)

        # Contextual predictions: predicting connectivity within the cell
        ctx_fcs = []
        for i in range(self.cell_num_layers - 1):
            # We substract 1 since the cell inputs are defined through connection predictions
            for _ in range(self._num_inputs_per_cell_layer):
                ctx_fcs.append(
                    # For each layer input we re-use it together the aggregated output of all
                    nn.Linear(
                        lstm_hidden_size,
                        self._num_inputs_per_cell
                        + (
                            self._num_inputs_per_cell_layer
                            + self._num_outputs_per_cell_layer
                        )
                        * i,
                    )
                )
        self.ctx_fcs = nn.ModuleList(ctx_fcs)

        # init parameters
        self.reset_parameters()

    def action_size(self):
        return self._action_len

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def sample(self):
        """Sample one architecture
        """
        return self.forward()

    def evaluate(self, action):
        """Evaluate entropy entropy and log probability of given architecture."""
        config = MicroController.action2config(
            action, dec_block=self.dec_num_cells, ctx_block=self.cell_num_layers
        )
        return self.forward(config)

    def forward(self, config=None):
        """Sample a decoder or compute the log_prob if decoder config is given

        Args:
          config (List): decoder architecture

        Returns:
          dec_arch: (ctx, conns)
            ctx: (index, op) x 4
            conns: [index_1, index_2] x 3
          entropy
          log_prob

        """
        do_sample = config is None
        if do_sample:
            ctx_config = []
            conns = []
        else:
            ctx_config, conns = config
        inputs = self.g_emb

        pool_hiddens = []
        hidden = (
            torch.zeros([self.lstm_num_layers, 1, self.lstm_hidden_size]),
            torch.zeros([self.lstm_num_layers, 1, self.lstm_hidden_size]),
        )
        entropy = 0
        log_prob = 0

        # Get enc layer hidden states
        for layer in range(self.enc_num_layers):
            output, hidden = self.rnn(inputs, hidden)
            pool_hiddens.append(output.squeeze())
            inputs = output

        # Sample connections
        for layer in range(self.dec_num_cells):
            if do_sample:
                conn = []
            for i in range(
                self._num_cells_per_decoder_block * self._num_inputs_per_cell
            ):
                output, hidden = self.rnn(inputs, hidden)
                logits = self.conn_fcs[
                    layer
                    * self._num_cells_per_decoder_block
                    * self._num_inputs_per_cell
                    + i * self._num_outputs_per_cell
                ](output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    index, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    conn.append(int(index))
                else:
                    index = torch_long(conns[layer][i])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, index)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            if do_sample:
                conns.append(conn)

        # sample contextual cell
        for layer in range(self.cell_num_layers):
            # sample position
            if layer == 0:
                pos = 0  # first position is always 0
                # sample operation
                output, hidden = self.rnn(inputs, hidden)
                logits = self.linear_op(output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                else:
                    op_id = torch_long(ctx_config[layer][1])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output

                if do_sample:
                    ctx_config.append(int(op_id))
            else:
                cfg = []
                # Sample position twice
                for i in range(self._num_inputs_per_cell_layer):
                    output, hidden = self.rnn(inputs, hidden)
                    logits = self.ctx_fcs[2 * layer - 2 + i](output.squeeze(0))
                    critic_logits = compute_critic_logits(logits)
                    if do_sample:
                        pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                        cfg.append(int(pos))
                    else:
                        pos = torch_long(ctx_config[layer][i])
                        curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
                    entropy += curr_ent
                    log_prob += curr_log_prob
                    inputs = output
                # Sample operation twice
                for i in range(
                    self._num_inputs_per_cell_layer * self._num_ops_per_cell_layer_input
                ):
                    output, hidden = self.rnn(inputs, hidden)
                    logits = self.linear_op(output.squeeze(0))
                    critic_logits = compute_critic_logits(logits)
                    if do_sample:
                        op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                        cfg.append(int(op_id))
                    else:
                        op_id = torch_long(ctx_config[layer][i + 2])
                        curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                    entropy += curr_ent
                    log_prob += curr_log_prob
                    inputs = output
                if do_sample:
                    ctx_config.append(cfg)
        return [ctx_config, conns], entropy, log_prob

    def evaluate_actions(self, actions_batch):
        log_probs, entropies = [], []
        action_length, action_size = actions_batch.shape
        for i in range(action_length):
            _, entropy, log_prob = self.evaluate(actions_batch[i])
            log_probs.append(log_prob.view(1))
            entropies.append(entropy.view(1))
        return torch.cat(log_probs), torch.cat(entropies)

    @staticmethod
    def get_mock():
        arc_seq = [
            [[0], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            [[0, 1], [2, 3], [4, 5]],
        ]
        entropy = 6
        log_prob = -1.4
        return arc_seq, entropy, log_prob

    @staticmethod
    def config2action(config):
        ctx, conns = config
        action = []

        for idx, cell in enumerate(ctx):
            if idx == 0:
                index = 0
                op = cell
                action += [index, op]
            else:
                index1, index2, op1, op2 = cell
                action += [index1, index2, op1, op2]

        for conn in conns:
            action += [conn[0], conn[1]]

        return action

    @staticmethod
    def action2config(action, enc_end=0, dec_block=3, ctx_block=4):
        ctx = []
        for i in range(ctx_block):
            if i == 0:
                # the first layer has only input + op
                ctx.append([action[i], action[i + 1]])
            else:
                # next layers have 2 inputs + 2 ops
                ctx.append(
                    [
                        action[(i - 1) * 4 + 2],
                        action[(i - 1) * 4 + 3],
                        action[(i - 1) * 4 + 4],
                        action[(i - 1) * 4 + 5],
                    ]
                )
        conns = []
        for i in range(dec_block):
            conns.append(
                [
                    action[4 * (ctx_block - 1) + 2 + i * 2],
                    action[4 * (ctx_block - 1) + 2 + i * 2 + 1],
                ]
            )
        return [ctx, conns]


class TemplateController(nn.Module):
    """Stacked LSTM-based controller for TemplateDecoder.

    First,
      generate the cells aka templates,
    then
      generate the structure:
        the structure is with repeats and with strides
        stride can be >1 for layers before floor(cell_num_layers / 2)
        stride can be only 1 for layers after that

    """

    def __init__(
        self,
        enc_num_layers,
        num_ops,
        num_agg_ops,
        lstm_hidden_size=100,
        lstm_num_layers=2,
        dec_num_cells=3,
        cell_num_layers=3,
        cell_max_repeat=4,
        cell_max_stride=2,
        **kwargs,
    ):
        """
        Args:
          enc_num_layers (int) : initial size of the sampling pool
          num_ops (int) : number of operations
          num_agg_ops (int) : number of aggregation operations
          lstm_hidden_size (int): number of hidden units of LSTM
          lstm_num_layers (int): number of LSTM layers
          dec_num_cells (int) : number of templates/cells
          cell_num_layers (int) : number of instantiations of templates
          cell_max_repeat (int) : max number of repeats of templates
          cell_max_stride (int) : max stride (power of 2)

        """
        super(TemplateController, self).__init__()

        # For each template we predict 3 values: op1, op2, op_agg
        self._num_ops_per_template = 3
        # Each template takes 2 inputs
        self._num_inputs_per_template = 2
        # Each template produces 1 output
        self._num_outputs_per_template = 1
        # For each layer we predict 5 values:
        # inp1, inp2, template, cell_max_repeat, stride
        self._num_actions_per_layer = 5

        # Additional configs
        self.enc_num_layers = enc_num_layers
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.dec_num_cells = dec_num_cells
        self.cell_num_layers = cell_num_layers

        # Controller
        self.rnn = nn.LSTM(lstm_hidden_size, lstm_hidden_size, lstm_num_layers)
        self.enc_op = nn.Embedding(num_ops, lstm_hidden_size)
        self.linear_op = nn.Linear(lstm_hidden_size, num_ops)
        self.linear_agg_op = nn.Linear(lstm_hidden_size, num_agg_ops)
        self.template_op = nn.Linear(lstm_hidden_size, self.dec_num_cells)
        self.repeat_op = nn.Linear(lstm_hidden_size, cell_max_repeat)
        self.stride_op = nn.Linear(lstm_hidden_size, cell_max_stride)
        self.dummy_stride_op = nn.Linear(
            lstm_hidden_size, 1
        )  # always predicting a single stride
        self.g_emb = nn.Parameter(torch.zeros(1, 1, lstm_hidden_size))

        self._action_len = (
            self._num_ops_per_template * self.dec_num_cells
            + self._num_actions_per_layer * self.cell_num_layers
        )

        # connections
        ctx_fcs = []
        for i in range(self.cell_num_layers):
            for j in range(self._num_inputs_per_template):
                ctx_fcs.append(
                    # Each time the sampling pool grows by i
                    nn.Linear(
                        lstm_hidden_size,
                        self.enc_num_layers + i * self._num_outputs_per_template,
                    )
                )
        self.ctx_fcs = nn.ModuleList(ctx_fcs)

        # init parameters
        self.reset_parameters()

    def action_size(self):
        return self._action_len

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def sample(self):
        return self.forward()

    def evaluate(self, action):
        config = TemplateController.action2config(
            action, dec_block=self.dec_num_cells, ctx_block=self.cell_num_layers,
        )
        return self.forward(config)

    def forward(self, config=None):
        """Sample a decoder or compute the log_prob if decoder config is given

        Config is representing the following decisions:
            [[op1, op2, agg_op]] * dec_num_cells
        and
            [[loc1, loc2, comb1, cell_max_repeat, stride]] * cell_num_layers

        Args:
          config (List): decoder architecture

        Returns:
          dec_arch
          entropy
          log_prob

        """
        do_sample = config is None
        if do_sample:
            dec_config = []
            str_config = []
        else:
            dec_config, str_config = config

        inputs = self.g_emb

        pool_hiddens = []
        hidden = (
            torch.zeros([self.lstm_num_layers, 1, self.lstm_hidden_size]),
            torch.zeros([self.lstm_num_layers, 1, self.lstm_hidden_size]),
        )
        entropy = 0
        log_prob = 0

        # Get enc layer hidden states
        for layer in range(self.enc_num_layers):
            output, hidden = self.rnn(inputs, hidden)
            pool_hiddens.append(output.squeeze())
            inputs = output

        # Sample templates
        for layer in range(self.dec_num_cells):
            cell = []
            # sample ops
            for i in range(self._num_ops_per_template):
                output, hidden = self.rnn(inputs, hidden)
                if i == self._num_ops_per_template - 1:
                    # Last op is the agg op
                    logits = self.linear_agg_op(output.squeeze(0))
                else:
                    logits = self.linear_op(output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    cell.append(int(op_id))
                else:
                    op_id = torch_long(dec_config[layer][i])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            if do_sample:
                dec_config.append(cell)
        # sample decoder structure
        for layer in range(self.cell_num_layers):
            cfg = []
            # sample locations
            for i in range(self._num_inputs_per_template):
                output, hidden = self.rnn(inputs, hidden)
                logits = self.ctx_fcs[layer * 2 + i](output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    pos, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    cfg.append(int(pos))
                else:
                    pos = torch_long(str_config[layer][i])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, pos)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            # Sample template index
            for i in range(1):
                output, hidden = self.rnn(inputs, hidden)
                logits = self.template_op(output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    cfg.append(int(op_id))
                else:
                    op_id = torch_long(str_config[layer][i + 2])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            # Sample number of repeats
            for i in range(1):
                output, hidden = self.rnn(inputs, hidden)
                logits = self.repeat_op(output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    cfg.append(int(op_id))
                else:
                    op_id = torch_long(str_config[layer][i + 3])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            # Sample stride
            for i in range(1):
                output, hidden = self.rnn(inputs, hidden)
                if layer >= (self.cell_num_layers // 2):
                    # We only predict stride after half of the layers were predicted
                    logits = self.dummy_stride_op(output.squeeze(0))
                else:
                    logits = self.stride_op(output.squeeze(0))
                critic_logits = compute_critic_logits(logits)
                if do_sample:
                    op_id, curr_ent, curr_log_prob = sample_logits(critic_logits)
                    cfg.append(int(op_id))
                else:
                    op_id = torch_long(str_config[layer][i + 4])
                    curr_ent, curr_log_prob = calc_prob(critic_logits, op_id)
                entropy += curr_ent
                log_prob += curr_log_prob
                inputs = output
            if do_sample:
                str_config.append(cfg)
        ctx_config = [dec_config, str_config]
        return ctx_config, entropy, log_prob

    def evaluate_actions(self, actions_batch):
        log_probs, entropies = [], []
        action_length, action_size = actions_batch.shape
        for i in range(action_length):
            _, entropy, log_prob = self.evaluate(actions_batch[i])
            log_probs.append(log_prob.view(1))
            entropies.append(entropy.view(1))
        return torch.cat(log_probs), torch.cat(entropies)

    @staticmethod
    def get_mock():
        enc_s = None
        arc_seq = [[[0, 0, 0]], [[0, 0, 0, 0, 0]]]
        entropy = 6
        log_prob = -1.4
        return enc_s, arc_seq, entropy, log_prob

    @staticmethod
    def config2action(config):
        decoder, structure = config
        action = []

        for cell in decoder:
            action += cell
        for block in structure:
            action += block
        return action

    @staticmethod
    def action2config(
        action,
        enc_end=0,
        dec_block=3,
        ctx_block=3,
        num_ops_per_template=3,
        num_actions_per_layer=5,
    ):
        decoder = []
        structure = []
        for i in range(dec_block):
            decoder.append(
                action[(i * num_ops_per_template) : (i + 1) * num_ops_per_template]
            )
        action = action[dec_block * num_ops_per_template :]
        for j in range(ctx_block):
            structure.append(
                action[(j * num_actions_per_layer) : (j + 1) * num_actions_per_layer]
            )
        return [decoder, structure]
