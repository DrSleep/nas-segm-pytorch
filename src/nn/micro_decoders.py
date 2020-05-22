"""NAS Decoders"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.genotypes import AGG_OP_NAMES, OP_NAMES, OP_NAMES_WACV
from .layer_factory import AGG_OPS, OPS, conv_bn_relu, conv3x3


def collect_all(feats, collect_indices):
    """Collect outputs from all layers into a single output"""
    out = feats[collect_indices[0]]
    for i in range(1, len(collect_indices)):
        collect = feats[collect_indices[i]]
        if out.size()[2] > collect.size()[2]:
            collect = nn.Upsample(
                size=out.size()[2:], mode="bilinear", align_corners=False
            )(collect)
        elif collect.size()[2] > out.size()[2]:
            out = nn.Upsample(
                size=collect.size()[2:], mode="bilinear", align_corners=False
            )(out)
        out = torch.cat([out, collect], 1)
    return out


class AggregateCell(nn.Module):
    """
        before aggregating, both paths should undergo
        conv1x1 with the output channels being equal to the smallest channel size among them
        upsampled to the highest resolution among them
        pre_transform: whether to do convbnrelu before summing up
    """

    def __init__(self, size_1, size_2, agg_size, pre_transform=True):
        super(AggregateCell, self).__init__()
        self.pre_transform = pre_transform
        if self.pre_transform:
            self.branch_1 = conv_bn_relu(size_1, agg_size, 1, 1, 0)
            self.branch_2 = conv_bn_relu(size_2, agg_size, 1, 1, 0)

    def forward(self, x1, x2):
        if self.pre_transform:
            x1 = self.branch_1(x1)
            x2 = self.branch_2(x2)
        if x1.size()[2:] > x2.size()[2:]:
            x2 = nn.Upsample(size=x1.size()[2:], mode="bilinear")(x2)
        elif x1.size()[2:] < x2.size()[2:]:
            x1 = nn.Upsample(size=x2.size()[2:], mode="bilinear")(x1)
        return x1 + x2


class ContextualCell(nn.Module):
    """New contextual cell design

    Config contains [op1, [loc1, loc2, op1, op2], [...], [...]]

    """

    def __init__(self, config, inp, repeats=1):
        super(ContextualCell, self).__init__()
        self._ops = nn.ModuleList()
        self._pos = []
        self._collect_inds = [0]
        self._pools = ["x"]
        for ind, op in enumerate(config):
            if ind == 0:
                # first op is always applied on x
                pos = 0
                op_id = op
                self._collect_inds.remove(pos)
                op_name = OP_NAMES[op_id]
                # turn-off scaling in batch norm
                self._ops.append(OPS[op_name](inp, inp, 1, True, repeats))
                self._pos.append(pos)
                self._collect_inds.append(ind + 1)
                self._pools.append("{}({})".format(op_name, self._pools[pos]))
            else:
                pos1, pos2, op_id1, op_id2 = op
                # drop op_id from loose ends
                for (pos, op_id) in zip([pos1, pos2], [op_id1, op_id2]):
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = OP_NAMES[op_id]
                    # turn-off scaling in batch norm
                    self._ops.append(OPS[op_name](inp, inp, 1, True, repeats))
                    self._pos.append(pos)
                    # Do not collect intermediate layers
                    self._pools.append("{}({})".format(op_name, self._pools[pos]))
                # summation
                op_name = "sum"
                # turn-off convbnrelu
                self._ops.append(
                    AggregateCell(
                        size_1=None, size_2=None, agg_size=inp, pre_transform=False
                    )
                )
                self._pos.append([ind * 3 - 1, ind * 3])
                self._collect_inds.append(ind * 3 + 1)
                self._pools.append(
                    "{}({},{})".format(
                        op_name, self._pools[ind * 3 - 1], self._pools[ind * 3]
                    )
                )

    def forward(self, x):
        feats = [x]
        for pos, op in zip(self._pos, self._ops):
            if isinstance(pos, list):
                assert len(pos) == 2, "Two ops must be provided"
                feats.append(op(feats[pos[0]], feats[pos[1]]))
            else:
                feats.append(op(feats[pos]))
        out = 0
        for i in self._collect_inds:
            out += feats[i]
        return out

    def prettify(self):
        return " + ".join(self._pools[i] for i in self._collect_inds)


class MergeCell(nn.Module):
    def __init__(self, ctx_config, conn, inps, agg_size, ctx_cell, repeats=1):
        super(MergeCell, self).__init__()
        self.index_1, self.index_2 = conn
        inp_1, inp_2 = inps
        self.op_1 = ctx_cell(ctx_config, inp_1, repeats=repeats)
        self.op_2 = ctx_cell(ctx_config, inp_2, repeats=repeats)
        self.agg = AggregateCell(inp_1, inp_2, agg_size)

    def forward(self, x1, x2):
        x1 = self.op_1(x1)
        x2 = self.op_2(x2)
        return self.agg(x1, x2)

    def prettify(self):
        return self.op_1.prettify()


class MicroDecoder(nn.Module):
    """
        Parent class for MicroDecoders
        l1, l2, l3, l4, None - pool of decision nodes

        Decoder config must include:
         cell config
         a list of aggregate positions (can be identical)

        in the end, all loose connections from modified layers
        must be aggregated via the concatenation operation
    """

    def __init__(
        self,
        inp_sizes,
        num_classes,
        config,
        agg_size=64,
        num_pools=4,
        ctx_cell=ContextualCell,
        aux_cell=False,
        repeats=1,
        **kwargs
    ):
        super(MicroDecoder, self).__init__()
        cells = []
        aux_clfs = []
        self.aux_cell = aux_cell
        self.collect_inds = []
        ## for description of the structure
        self.pool = ["l{}".format(i + 1) for i in range(num_pools)]
        self.info = []
        self.agg_size = agg_size

        ## NOTE: bring all outputs to the same size
        for out_idx, size in enumerate(inp_sizes):
            setattr(
                self,
                "adapt{}".format(out_idx + 1),
                conv_bn_relu(size, agg_size, 1, 1, 0, affine=True),
            )
            inp_sizes[out_idx] = agg_size

        inp_sizes = inp_sizes.copy()
        cell_config, conns = config
        self.conns = conns
        self.ctx = cell_config
        self.repeats = repeats
        self.collect_inds = []
        self.ctx_cell = ctx_cell
        for block_idx, conn in enumerate(conns):
            for ind in conn:
                if ind in self.collect_inds:
                    # remove from outputs if used by pool cell
                    self.collect_inds.remove(ind)
            ind_1, ind_2 = conn
            cells.append(
                MergeCell(
                    cell_config,
                    conn,
                    (inp_sizes[ind_1], inp_sizes[ind_2]),
                    agg_size,
                    ctx_cell,
                    repeats=repeats,
                )
            )
            aux_clfs.append(nn.Sequential())
            if self.aux_cell:
                aux_clfs[block_idx].add_module(
                    "aux_cell", ctx_cell(self.ctx, agg_size, repeats=repeats)
                )
            aux_clfs[block_idx].add_module(
                "aux_clf", conv3x3(agg_size, num_classes, stride=1, bias=True)
            )
            self.collect_inds.append(block_idx + num_pools)
            inp_sizes.append(agg_size)
            ## for description
            self.pool.append("({} + {})".format(self.pool[ind_1], self.pool[ind_2]))
        self.cells = nn.ModuleList(cells)
        self.aux_clfs = nn.ModuleList(aux_clfs)
        self.pre_clf = conv_bn_relu(
            agg_size * len(self.collect_inds), agg_size, 1, 1, 0
        )
        self.conv_clf = conv3x3(agg_size, num_classes, stride=1, bias=True)
        self.info = " + ".join(self.pool[i] for i in self.collect_inds)
        self.num_classes = num_classes

    def prettify(self, n_params):
        """ Encoder config: None
            Dec Config:
              ctx: (index, op) x 4
              conn: [index_1, index_2] x 3
        """
        header = "#PARAMS\n\n {:3.2f}M".format(n_params / 1e6)
        ctx_desc = "#Contextual:\n" + self.cells[0].prettify()
        conn_desc = "#Connections:\n" + self.info
        return header + "\n\n" + ctx_desc + "\n\n" + conn_desc

    def forward(self, x):
        x = list(x)
        aux_outs = []
        for out_idx in range(len(x)):
            x[out_idx] = getattr(self, "adapt{}".format(out_idx + 1))(x[out_idx])
        for cell, aux_clf, conn in zip(self.cells, self.aux_clfs, self.conns):
            cell_out = cell(x[conn[0]], x[conn[1]])
            x.append(cell_out)
            aux_outs.append(aux_clf(cell_out.clone()))
        out = collect_all(x, self.collect_inds)
        out = F.relu(out)
        out = self.pre_clf(out)
        out = self.conv_clf(out)
        return out, aux_outs


class TemplateDecoder(nn.Module):
    """
        Parent class for TemplateDecoders with template-repeats and strides
    """

    def __init__(
        self,
        inp_sizes,
        num_classes,
        config,
        agg_size=64,
        num_pools=4,
        repeats=1,
        stride_power=1,
        **kwargs
    ):
        """
        Args:
            stride_power (int, default=1): multiply the number of channels
                by int(stride**stride_power)

        """

        super(TemplateDecoder, self).__init__()

        inp_sizes = inp_sizes.copy()
        n_scales = len(inp_sizes)
        cells, structure = config
        n_blocks = len(structure)
        # keep track of the channel dimension
        _channels = inp_sizes + [0] * n_blocks
        # whether to reduce or increase the number of channels
        agg_fun = max

        self._ops = nn.ModuleList()
        self._pos = []  # at which position to apply op
        self._collect_inds = []  # collect only the last layer
        self._repeats = []  # how many times to repeat the template
        self._pools = ["l{}".format(j + 1) for j in range(n_scales)]

        for block_idx, (pos1, pos2, cell_id, num_repeats, stride) in enumerate(
            structure
        ):
            # we only do down-sampling in the first half of the layers
            larger = block_idx >= (len(structure) // 2)
            num_repeats += 1  # as it is zero-indexed
            stride = 2 ** stride
            op_id1, op_id2, op_agg = cells[cell_id]
            _ops = nn.ModuleList()
            _pos = []
            agg_channel = None
            new_channels = [0, 0]
            prev_channels = [0, 0]
            for repeat_idx in range(num_repeats):
                for layer_idx, (pos, op_id) in enumerate(
                    zip([pos1, pos2], [op_id1, op_id2])
                ):
                    if repeat_idx == 0:
                        curr_channel = _channels[pos]
                        new_channel = curr_channel * int(stride ** stride_power)
                    elif layer_idx == 0:
                        curr_channel = new_channel = prev_channels[-1]
                    else:
                        curr_channel = new_channel = agg_channel
                    new_channels[layer_idx] = new_channel
                    prev_channels[layer_idx] = curr_channel
                    if pos in self._collect_inds:
                        self._collect_inds.remove(pos)
                    op_name = OP_NAMES_WACV[op_id]
                    _ops.append(
                        OPS[op_name](
                            curr_channel, new_channel, stride, True, repeats=repeats
                        )
                    )
                    _pos.append(pos)
                    self._pools.append("{}({})".format(op_name, self._pools[pos]))
                # aggregation
                op_name = AGG_OP_NAMES[op_agg]
                agg_channel = agg_fun(new_channels)
                _ops.append(
                    AGG_OPS[op_name](
                        new_channels[0],
                        new_channels[1],
                        agg_channel,
                        True,
                        repeats=repeats,
                        larger=larger,
                    )
                )
            collect_idx = n_scales + block_idx
            _channels[collect_idx] = agg_channel
            self._pos.append(_pos)
            self._ops.append(_ops)
            self._repeats.append(num_repeats)
            self._collect_inds.append(collect_idx)
            self._pools.append(
                "{}({},{})".format(
                    op_name,
                    self._pools[len(inp_sizes) + block_idx - 2],
                    self._pools[len(inp_sizes) + block_idx - 1],
                )
            )
        c_pre_clf = sum(
            [c for idx, c in enumerate(_channels) if idx in self._collect_inds]
        )
        self.pre_clf = conv_bn_relu(c_pre_clf, agg_size, 1, 1, 0)
        self.conv_clf = conv3x3(agg_size, num_classes, stride=1, bias=True)
        self.info = " + ".join(self._pools[i] for i in self._collect_inds)
        self.num_classes = num_classes

    def _reset_clf(self, num_classes):
        if num_classes != self.num_classes:
            del self.conv_clf
            self.conv_clf = conv3x3(
                self.agg_size, num_classes, stride=1, bias=True
            ).cuda()
            self.num_classes = num_classes

    def prettify(self, n_params):
        header = "#PARAMS\n\n {:3.2f}M".format(n_params / 1e6)
        conn_desc = "#Connections:\n" + self.info
        return header + "\n\n" + conn_desc

    def forward(self, x):
        feats = list(x)
        for pos, ops, repeat in zip(self._pos, self._ops, self._repeats):
            assert isinstance(pos, list), "Must be list"
            loc1, loc2 = pos[:2]
            feat1, feat2 = feats[loc1], feats[loc2]
            for i in range(repeat):
                out0 = ops[i * 3](feat1)
                out1 = ops[i * 3 + 1](feat2)
                out2 = ops[i * 3 + 2](out0, out1)
                if i < (repeat - 1):
                    feat1 = feat2.clone()
                    feat2 = out2.clone()
            feats.append(out2)
        out = collect_all(feats, self._collect_inds)
        out = F.relu(out)
        out = self.pre_clf(out)
        out = self.conv_clf(out)
        return out
