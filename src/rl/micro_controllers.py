"""RL Controller"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_long(x):
    x = torch.tensor(x, dtype=torch.long)
    return x.view((1, 1))


class MicroController(nn.Module):
    """ Stack LSTM controller based on ENAS

    https://arxiv.org/abs/1802.03268

    With modification that indices and ops chosen with linear classifier
    Samples output strides for encoder and decoder structure
    """

    def __init__(
        self,
        num_enc_scales,
        op_size,
        hidden_size=100,
        num_lstm_layers=2,
        num_dec_layers=3,
        num_ctx_layers=4,
    ):
        """
        Args:
          num_enc_scales (int): encoder input scales
          op_size (int): numebr of operations
          hidden_size (int): number of hidden units of LSTM
          num_lstm_layers (int): number of LSTM layers
          num_dec_layers (int): number of cells in the decoder
          num_ctx_layers (int): numebr of layers in each cell
        """
        super(MicroController, self).__init__()

        # additional configs
        self.num_enc_scales = num_enc_scales
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.num_dec_layers = num_dec_layers
        self.num_ctx_layers = num_ctx_layers
        self.temperature = None
        self.tanh_constant = None

        # the network
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_lstm_layers)
        self.enc_op = nn.Embedding(op_size, hidden_size)
        self.linear_op = nn.Linear(hidden_size, op_size)
        self.g_emb = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self._action_len = (
            2 * num_dec_layers + 4 * (num_ctx_layers - 1) + 2
        )  # 2 for first 2 in ctx

        # connection predictions
        conn_fcs = []
        for l in range(num_dec_layers):
            for _ in range(2):
                conn_fcs.append(nn.Linear(hidden_size, num_enc_scales + l))
        self.conn_fcs = nn.ModuleList(conn_fcs)

        # contextual predictions
        ctx_fcs = []
        for l in range(2, num_ctx_layers + 1):
            for _ in range(2):
                ctx_fcs.append(
                    nn.Linear(hidden_size, l * 3 - 4)
                )  # for 2 = 2, 3 = 5, 4 = 8, etc.
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
            action, dec_block=self.num_dec_layers, ctx_block=self.num_ctx_layers
        )
        return self.forward(config)

    def forward(self, config=None):
        """ sample a decoder or compute the log_prob if decoder config is given
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
            dec_arch = config
            ctx_config, conns = dec_arch
        inputs = self.g_emb

        pool_hiddens = []
        hidden = (
            torch.zeros([self.num_lstm_layers, 1, self.hidden_size]),
            torch.zeros([self.num_lstm_layers, 1, self.hidden_size]),
        )
        entropy = 0
        log_prob = 0

        def calc_prob(critic_logits, x):
            """compute entropy and log_prob."""
            softmax_logits, log_softmax_logits = critic_logits
            ent = softmax_logits * log_softmax_logits
            ent = -1 * ent.sum()
            log_prob = -F.nll_loss(log_softmax_logits, x.view(1))
            return ent, log_prob

        def compute_critic_logits(logits):
            softmax_logits = F.softmax(logits, dim=-1)
            log_softmax_logits = F.log_softmax(logits, dim=-1)
            critic_logits = (softmax_logits, log_softmax_logits)
            return critic_logits

        def sample_logits(critic_logits):
            softmax_logits = critic_logits[0]
            x = softmax_logits.multinomial(num_samples=1)
            ent, log_prob = calc_prob(critic_logits, x)
            return x, ent, log_prob

        # get enc layer hidden states
        for layer in range(self.num_enc_scales):
            output, hidden = self.rnn(inputs, hidden)
            pool_hiddens.append(output.squeeze())
            # don't sample stride config
            inputs = output

        # sample connections
        for layer in range(self.num_dec_layers):
            if do_sample:
                conn = []
            # sample index_1, index_2
            for i in range(2):
                output, hidden = self.rnn(inputs, hidden)
                logits = self.conn_fcs[layer * 2 + i](output.squeeze(0))
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
        for layer in range(self.num_ctx_layers):
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
                # sample position twice
                for i in range(2):
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
                # sample operation twice
                for i in range(2):
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
        dec_arch = config
        ctx, conns = dec_arch
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
                ctx.append([action[i], action[i + 1]])
            else:
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
