"""局部 Actor 的观测编码器，以及几种信息范围不同的 Critic 网络。

阅读本文件时，可以先分清三个职责：
1. Encoder 把环境观测转换为神经网络特征。
2. ActorModule 把自己的局部特征转换为动作分布的参数。
3. Critic 估计收益：Q 网络同时读取观测和动作，V 网络只读取观测。

这里定义的是网络结构和前向计算。奖励设计、经验采样、损失函数、
优化器更新以及目标网络的软更新，由环境和各算法的 Learner 负责。
共享 Critic 或共享注意力不等于共享奖励，也不会自动把任务变成合作任务。

下文统一使用这些张量形状符号：
    B：一个训练批次中的样本数；每个样本对应一条联合环境转换。
    N：智能体数量。
    D：hidden，也就是网络的隐藏特征宽度。
    C_i：智能体 i 的当前观测展开后的维数。
    A_i：智能体 i 的动作编码特征维数。
    T：历史窗口长度；F：每条历史记录的特征维数。

例如 B=128、N=3、D=64 时，注意力编码 e 的形状为 [128, 3, 64]，
注意力权重为 [128, 3, 3]，最终各智能体的 Q 值为 [128, 3]。
"""

import copy  # 用 deepcopy 创建目标策略，避免与在线策略引用同一组参数。

import torch
from ray.rllib.core.columns import Columns  # RLlib 对 obs、动作分布参数等数据列的统一命名。
from ray.rllib.core.rl_module.apis.inference_only_api import InferenceOnlyAPI  # 声明哪些网络只用于训练。
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule  # RLlib 的 PyTorch 模型接口。
from torch import nn

from raymarl.distributions import HybridDistribution, bound_distribution  # 混合动作分布及其绑定类。
from raymarl.spaces import ActionLayout, ObservationLayout  # 环境空间与网络张量之间的转换规则。


class Encoder(nn.Module):
    """编码一个智能体的观测，可同时使用当前状态与一段历史。

    space 是该编码器实际接收的 Gymnasium 观测空间：Actor 通常是局部空间，
    集中 Value 的编码器则可以接收联合观测空间。

    project=True：最后用两层 MLP 投影到固定宽度 D。
    project=False：不做最后的 MLP 投影，保留当前特征及历史编码的拼接。
    注意力 Critic 使用后一种方式，再由自己的状态-动作投影 g_i 处理。
    """

    def __init__(self, space, hidden, project=True):
        super().__init__()
        # ObservationLayout 负责确定当前观测的展开长度、是否存在 history，
        # 以及如何把原始观测组织成 current/history/history_mask 三个字段。
        self.layout = ObservationLayout(space)
        # 只有声明了历史窗口时才创建 LSTM。
        # history_shape=[T, F]，所以 history_shape[1] 是 LSTM 的输入宽度 F。
        # batch_first=True 表示输入是 [B, T, F]，输出是 [B, T, D]。
        self.history_lstm = (
            nn.LSTM(self.layout.history_shape[1], hidden, batch_first=True)
            if self.layout.history_shape
            else None
        )
        # 无历史：特征宽度为 C_i。
        # 有历史：把当前观测 [B, C_i] 与历史摘要 [B, D] 拼接，宽度为 C_i+D。
        width = self.layout.current_dim + (hidden if self.history_lstm else 0)
        # 对外声明这个编码器实际输出多少个特征，便于后续 Linear 设置输入尺寸。
        self.feature_dim = hidden if project else width
        # project=True：width -> D -> D，每层后使用 LeakyReLU 引入非线性。
        # project=False：Identity 原样返回输入；之前的历史 LSTM 仍然可以有参数。
        self.output = (
            nn.Sequential(nn.Linear(width, hidden), nn.LeakyReLU(), nn.Linear(hidden, hidden), nn.LeakyReLU())
            if project
            else nn.Identity()
        )

    def forward(self, obs):
        """接收已经整理好的张量字典，输出 [B, feature_dim]。

        obs['current']：[B, C_i]。
        有历史时，obs['history']：[B, T, F]，obs['history_mask']：[B, T]。
        mask 必须与历史记录对应，采用有效记录在前、零填充在后的约定。
        这些张量的设备应与网络参数一致。
        """
        # 从当前观测开始；它可以包含 SOC、温度、时间、电价等已整理好的数值。
        x = obs["current"]
        if self.history_lstm:
            # h 保存历史序列每个时间位置的输出：[B, T, D]。
            # 第二个返回值包含最终 hidden/cell state；本实现不跨调用保存它们。
            # 因此，这里是“每次编码一个历史窗口”，不是跨 step 持久维护隐藏状态。
            h, _ = self.history_lstm(obs["history"])
            mask = obs["history_mask"]
            # 每个样本的有效历史长度为 mask.sum(-1)。
            # 例如 [1, 1, 0, 0] 的长度为 2，最后一条有效记录的下标为 1。
            # 无有效历史时，长度为 0；先将下标夹到 0，避免使用 -1 索引。
            index = (mask.sum(-1).long() - 1).clamp_min(0)
            # 对每个批次样本分别取它的最后一个有效历史输出，得到 [B, D]。
            # 这里按 mask 选取输出；LSTM 本身仍计算了整个窗口，包括填充位置。
            h = h[torch.arange(h.shape[0], device=h.device), index]
            # 无历史的样本把摘要清零。否则 LSTM 的偏置也可能产生非零输出。
            # unsqueeze(-1)：[B] -> [B, 1]，用于向 D 个特征广播乘法。
            h = h * (mask.sum(-1) > 0).unsqueeze(-1)
            # cat 沿已有的最后一维拼接：[B, C_i] + [B, D] -> [B, C_i+D]。
            x = torch.cat((x, h), -1)
        # 输出固定宽度特征，或者在 project=False 时直接返回拼接后的特征。
        return self.output(x)


class ActorModule(TorchRLModule, InferenceOnlyAPI):
    """通用局部随机策略模块：观测 -> 特征 -> 动作分布参数。

    每个智能体通常单独构造一个 Actor，因而默认拥有自己的网络参数。
    局部执行的信息边界取决于传入的 observation_space 和观测内容；
    本类不会自动从联合观测中删除其他智能体的信息。

    混合动作的具体采样、概率密度及动作解码，由 HybridDistribution 处理。
    """

    def setup(self):
        # setup 是 RLlib 建立模型时调用的初始化入口。
        hidden = self.model_config.get("hidden_dim", 64)
        # 默认 project=True，因此编码器输出为 [B, D]。
        self.encoder = Encoder(self.observation_space, hidden)
        # ActionLayout 记录各动作分支的空间、分布参数数目和 Critic 特征数目。
        self.layout = ActionLayout(self.action_space)
        # head 输出的是“分布参数”，并非已经采样出来的动作。
        # 例如一维连续动作需要 mean 和 log_std 两个数；一个二元动作需要一个 logit。
        # param_dim 与动作特征 feature_dim 不一定相同。
        self.head = nn.Linear(hidden, self.layout.param_dim)

    def distribution(self, obs, *, target=False):
        """给定 Encoder 所需的张量字典，返回可采样、可计算 log-prob 的分布实例。

        target=True 的调用方必须已经通过 make_target_policy 创建目标策略。
        各算法决定是否需要目标 Actor，本基类不会在 setup 中自动创建它。
        """
        if target:
            # 使用独立的目标编码器和目标输出头，而不是当前正在优化的在线参数。
            logits = self.target_policy["head"](self.target_policy["encoder"](obs))
        else:
            logits = self.head(self.encoder(obs))
        # logits 是这里对整个参数向量的统称；连续分支中还包含 mean/log_std。
        # 分支如何拆开、连续动作如何变换到边界内，交给分布类完成。
        return HybridDistribution(logits, action_space=self.action_space)

    def _forward(self, batch, **kwargs):
        """RLlib 的通用前向入口，返回动作分布的输入参数 [B, param_dim]。

        输入的 batch[Columns.OBS] 是 RLlib 连接器整理后的原始观测张量结构；
        torch_batch 再转换成 Encoder 需要的 current/history/history_mask。
        这个方法只计算参数；动作采样由相应分布和 RLlib 后续流程处理。
        """
        return {
            Columns.ACTION_DIST_INPUTS: self.head(
                self.encoder(self.encoder.layout.torch_batch(batch[Columns.OBS]))
            )
        }

    def get_train_action_dist_cls(self):
        # 返回“绑定了当前动作空间的分布类”，不是某一批样本的分布实例。
        # RLlib 随后用该类和 _forward 输出的参数构造实际分布。
        return bound_distribution(self.action_space)

    # 通用 Actor 在训练、探索、推理中使用相同的分布类型。
    # 使用相同类型不代表每种阶段都随机采样；确定性行为由相应调用流程决定。
    get_inference_action_dist_cls = get_train_action_dist_cls
    get_exploration_action_dist_cls = get_train_action_dist_cls

    def get_non_inference_attributes(self):
        # InferenceOnlyAPI 用这个列表过滤只服务训练的属性及其参数。
        # 执行动作只需要在线 Actor，所以目标策略不进入推理权重。
        return ["target_policy"]

    def make_target_policy(self):
        # deepcopy 保证目标策略与在线策略初始数值相同，但参数对象彼此独立。
        # ModuleDict 将两个子网络注册为 PyTorch 模块，支持设备迁移和状态保存。
        # requires_grad_(False) 禁止通过反向传播训练目标参数；
        # 目标软更新由各算法在其他文件中显式执行，这里不进行更新。
        self.target_policy = nn.ModuleDict(
            {"encoder": copy.deepcopy(self.encoder), "head": copy.deepcopy(self.head)}
        ).requires_grad_(False)

    def online_parameters(self):
        # 优化器只取得在线 encoder/head 的参数，避免把目标策略也放进优化器。
        return list(self.encoder.parameters()) + list(self.head.parameters())


class AttentionCritic(nn.Module):
    """Ye 2021 路线的单头注意力 Critic，为每个智能体输出自己的 Q_i。

    对智能体 i：
        e_i = g_i(观测特征_i, 动作特征_i)
        x_i = sum_{j != i}(weight_ij * value_j)
        Q_i = f_i(e_i, x_i)

    g_i 和 f_i 按智能体分别构造；Wq/Wk/Wv 在所有智能体间共享。
    “共享”指同一 Critic 内各智能体位置使用同一组注意力变换参数，
    不表示 Actor 共享观测或所有智能体必须使用相同奖励。
    """

    def __init__(self, observation_spaces, action_spaces, hidden=64):
        super().__init__()
        # observation_spaces/action_spaces 的列表顺序就是智能体顺序，后续输入必须一致。
        self.n = len(observation_spaces)
        # 每个智能体先整理自己的观测，有历史时使用各自的 LSTM。
        # project=False 保留当前特征和历史摘要，再交给下面的 g_i 做一层投影。
        # 因此即使各智能体的 C_i 不同，后续也可以统一映射为 D 维。
        self.encoders = nn.ModuleList(Encoder(s, hidden, project=False) for s in observation_spaces)
        # 每个 g_i 独立构造：[B, C_i(或 C_i+D)+A_i] -> [B, D]。
        # 动作已由 ActionLayout 编码为网络特征，而不是原始环境动作字典。
        self.action_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(e.feature_dim + ActionLayout(a).feature_dim, hidden), nn.LeakyReLU())
            for e, a in zip(self.encoders, action_spaces)
        )
        # 注意：这里的 q 是 query（查询向量）变换，不是最终的 Q-value 网络。
        # q_i 表示“智能体 i 根据自身观测-动作特征，需要关注怎样的信息”。
        self.q = nn.Linear(hidden, hidden, bias=False)
        # key_j 表示“智能体 j 的信息如何与别人的查询进行匹配”。
        self.k = nn.Linear(hidden, hidden, bias=False)
        # value_j 是匹配之后真正被加权汇总的信息。这里额外经过 LeakyReLU。
        self.v = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.LeakyReLU())
        # 每个智能体有自己的 Q 输出头 f_i。
        # 输入是自身 e_i 与其他智能体摘要 x_i，各 D 维，所以总宽度为 2D。
        # 最后一层没有激活函数，允许 Q 值为任意正负数。
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))
            for _ in observation_spaces
        )

    def encode_observations(self, observations):
        # observations 是长度为 N 的列表；第 i 项是第 i 个智能体的观测张量字典。
        # 返回同顺序的特征列表，每项为 [B, enc.feature_dim]。
        return [enc(o) for enc, o in zip(self.encoders, observations)]

    def forward(self, observations, actions, return_attention=False, encoded=None):
        """计算联合批次中每个智能体的 Q 值。

        observations：N 个观测字典，批次维必须相同且样本时间身份对齐。
        actions：N 个 [B, A_i] 动作特征张量。
        encoded：可选的观测编码缓存，通常用于固定观测、改变候选动作的计算。
                 缓存只省去观测编码；状态-动作编码和注意力仍随动作重新计算。
        return_attention=True 时同时返回注意力矩阵，便于查看关注关系。
        """
        encoded = self.encode_observations(observations) if encoded is None else encoded
        # 第一步：拼接每个智能体的观测与动作，再由它自己的 g_i 映射成 D 维。
        # stack(..., 1) 新建智能体维：N 个 [B, D] -> 一个 [B, N, D]。
        # 与 cat 不同，stack 要求这里每项尺寸一致；前面的 g_i 已完成宽度统一。
        e = torch.stack(
            [g(torch.cat((o, a), -1)) for g, o, a in zip(self.action_encoders, encoded, actions)], 1
        )
        # weights[b, i, j] 表示第 b 个样本中，智能体 i 对智能体 j 的注意力。
        # new_zeros 继承 e 的 dtype/device，避免 CPU/GPU 张量混用。
        weights = e.new_zeros((e.shape[0], self.n, self.n))
        # context 是每个智能体收到的其他智能体信息摘要：[B, N, D]。
        context = torch.zeros_like(e)
        if self.n > 1:
            # query/key：[B, N, D]；key 转置后：[B, D, N]。
            # 矩阵乘法为每个样本生成所有 i-j 的匹配分数：[B, N, N]。
            # 本实现按能源论文路线保留未缩放的单头点积，没有除以 sqrt(D)。
            logits = self.q(e) @ self.k(e).transpose(-2, -1)
            # 对角线为 True，表示 j=i 的位置，也就是“关注自己”。
            # mask 的 [N, N] 形状会自动广播到整个批次 [B, N, N]。
            mask = torch.eye(self.n, device=e.device, dtype=torch.bool)
            # 自身位置设为 -inf 后，softmax 权重为 0。
            # softmax(-1) 沿信息来源 j 归一化，每个查询 i 的其他智能体权重之和为 1。
            weights = logits.masked_fill(mask, -torch.inf).softmax(-1)
            # [B, N, N] @ [B, N, D] -> [B, N, D]，实现其他智能体信息的加权和。
            # 例如 weights[b, i]=[0, 0.7, 0.3]，则摘要为 0.7*value_1+0.3*value_2。
            context = weights @ self.v(e)
        # N=1 时没有其他智能体，直接使用零摘要和零注意力。
        # 这样也避免“唯一位置被屏蔽为 -inf 后再 softmax”产生 NaN。
        # 最后分别计算每个 Q_i：[B, 2D] -> [B, 1] -> [B]。
        # squeeze(-1) 只去掉末尾的单元素维；最后 stack 得到 [B, N]。
        result = torch.stack(
            [head(torch.cat((e[:, i], context[:, i]), -1)).squeeze(-1) for i, head in enumerate(self.heads)],
            -1,
        )
        # 注意力压缩了每个 Q 头的输入；全连接注意力矩阵仍包含 N*N 个位置，
        # 所以不能据此认为总计算量随智能体数量线性增长。
        return (result, weights) if return_attention else result


class JointQCritic(nn.Module):
    """集中式联合 Q：每个输出头都读取所有智能体的观测特征与动作。

    与 AttentionCritic 的区别是这里直接拼接，不计算注意力摘要。
    N 个输出头分别估计 Q_i(联合观测, 联合动作)，收益目标由各自奖励决定。
    """

    def __init__(self, observation_spaces, action_spaces, hidden=64):
        super().__init__()
        # 每个观测编码器默认投影到固定 D 维，支持各智能体原始观测尺寸不同。
        self.encoders = nn.ModuleList(Encoder(s, hidden) for s in observation_spaces)
        # 联合输入宽度为 sum_i(D+A_i)。动作维数可以因智能体而异。
        width = sum(e.feature_dim + ActionLayout(a).feature_dim for e, a in zip(self.encoders, action_spaces))
        # 输出头分别构造，输入同一个联合向量，但可以学习不同智能体的收益。
        # 每个输出头的结构：width -> D -> D -> 1。
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, hidden),
                nn.LeakyReLU(),
                nn.Linear(hidden, hidden),
                nn.LeakyReLU(),
                nn.Linear(hidden, 1),
            )
            for _ in observation_spaces
        )

    def encode_observations(self, observations):
        # 返回 N 个 [B, D] 特征；这个 Critic 内的各 Q 头复用这些编码结果。
        return [e(o) for e, o in zip(self.encoders, observations)]

    def forward(self, observations, actions, encoded=None):
        """按构造时的智能体顺序接收联合批次，输出 [B, N]。

        actions[i] 是 [B, A_i]。所有观测和动作必须来自同一批联合时间步。
        可选 encoded 缓存必须对应当前观测，不能混入其他批次的编码。
        """
        encoded = self.encode_observations(observations) if encoded is None else encoded
        # zip 先形成 (观测特征_0, 动作_0)、(观测特征_1, 动作_1) 等配对。
        # 列表推导展开为 [观测_0, 动作_0, 观测_1, 动作_1, ...]，
        # 然后 cat 沿最后一维拼接成 [B, sum_i(D+A_i)]。
        x = torch.cat([v for pair in zip(encoded, actions) for v in pair], -1)
        # 各头各输出一个 [B]，再新建末尾智能体维，得到 [B, N]。
        return torch.stack([h(x).squeeze(-1) for h in self.heads], -1)


class TwinJointQCritic(nn.Module):
    """两套参数独立的联合 Q 网络，用于联合双 Q SAC。

    双 Q 指两个同时学习的在线估计 Q1/Q2，不是“在线网络与目标网络”。
    各自的目标副本在其他模块中建立；取两者最小值等训练规则由 Learner 执行。
    """

    def __init__(self, observation_spaces, action_spaces, hidden=64):
        super().__init__()
        # 两次构造意味着编码器和 Q 头都不共享参数，结构及输入尺寸相同。
        self.q1 = JointQCritic(observation_spaces, action_spaces, hidden)
        self.q2 = JointQCritic(observation_spaces, action_spaces, hidden)

    def forward(self, observations, actions):
        # 返回一个二元 tuple，每项都是 [B, N]；这里不平均、不求和、不取最小值。
        return self.q1(observations, actions), self.q2(observations, actions)


class LocalQCritic(nn.Module):
    """局部动作价值网络 Q_i(o_i, a_i)，只处理一个智能体的输入。

    这里的局部指实际传入的信息范围。若上层把全局信息放进 observation，
    本网络不会自动检测或排除它；信息边界应由环境及模型接线保证。
    """

    def __init__(self, observation_space, action_space, hidden=64):
        super().__init__()
        # 局部观测 -> [B, D]；它与 Actor 的 Encoder 是分别构造的网络。
        self.encoder = Encoder(observation_space, hidden)
        # 拼接本智能体的动作后，宽度为 D+A_i；输出一个不受正负限制的 Q 值。
        self.head = nn.Sequential(
            nn.Linear(hidden + ActionLayout(action_space).feature_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observation, action):
        # [B, D] 与 [B, A_i] 拼接，送入 Q 头；squeeze 后返回 [B]。
        # 此处没有其他智能体的观测或动作参数。
        return self.head(torch.cat((self.encoder(observation), action), -1)).squeeze(-1)


class TwinLocalQCritic(nn.Module):
    """独立 SAC 使用的两套局部 Q 网络；每套都有自己的观测编码器和输出头。"""

    def __init__(self, observation_space, action_space, hidden=64):
        super().__init__()
        # Q1/Q2 参数独立。各智能体通常还会分别拥有自己的 TwinLocalQCritic。
        self.q1 = LocalQCritic(observation_space, action_space, hidden)
        self.q2 = LocalQCritic(observation_space, action_space, hidden)

    def forward(self, observation, action):
        # 返回 (Q1, Q2)，两项形状都为 [B]。SAC 的最小值计算位于学习器中。
        return self.q1(observation, action), self.q2(observation, action)


class LocalValue(nn.Module):
    """状态价值网络 V(o)，在给定观测下估计策略后续的累计收益。

    与 Q 网络不同，它不接收一个待评价的动作。PPO 的价值回归和 GAE
    使用这类输出，但对应的损失和优势计算不在本文件中。
    """

    def __init__(self, space, hidden):
        super().__init__()
        # layout 把 RLlib 的原始观测张量整理成 Encoder 使用的字段。
        self.layout = ObservationLayout(space)
        # Value 的编码器单独构造；它与 Actor 的编码器不共享参数对象。
        self.encoder = Encoder(space, hidden)
        # [B, D] -> [B, 1]，最后不加激活函数，允许正负价值估计。
        self.head = nn.Linear(hidden, 1)

    def forward(self, observation):
        # 此处接收原始张量观测结构，而不是已经编码好的特征。
        # 先按空间规则整理，再编码和计算 Value，最终返回 [B]。
        return self.head(self.encoder(self.layout.torch_batch(observation))).squeeze(-1)


class CentralValue(LocalValue):
    """标记使用集中信息的 Value 类型，前向计算直接继承 LocalValue。

    这个类没有重写网络结构，也不会自己收集其他智能体的观测。
    上层 MAPPO 模块把训练用的联合 critic 观测交给它，因此形成 V(联合观测)。
    IPPO 则把局部观测交给 LocalValue，形成 V_i(o_i)。

    类型命名用来明确预期的信息范围；真正的范围由传入的 space 和数据决定。
    """
