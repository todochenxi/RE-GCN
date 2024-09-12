## LSPE

- h, p, e的更新
- cat(src[h], p) * norm + wh
- batch_norm h,e
- 进行残差连接

## Bi-GCN

- GCN 的每一层都涉及来自源帖子的信息，以增强谣言根源的影响。
- 基于 CNN 的方法可以获得局部邻居内的相关特征，但无法处理图或树中的全局结构关系（Bruna 等人，2014）。 因此，这些方法忽略了谣言传播的全局结构特征。 实际上，CNN 并不是为了从结构化数据中学习高级表示而设计的，但图卷积网络 (GCN) 却是这样的（Kipf 和 Welling 2017）。
- x->gcn->h
- x->f(x)->xroot
- h^ = cat(h, xroot)
- 然后反过来再做一遍
- xroot 是 根特征的映射，和新获取的表示连接后，可以增强谣言根源的影响。
- 迁移：在tkg中，并不存在根特征，也无法针对性的增强
- 创造根特征，节点在不同时间步的频率一定程度上可以代表它的影响性，需要一种模型去选择出根特征，或者一种采样策略。。。(参考berthttps://zh.d2l.ai/chapter_natural-language-processing-pretraining/bert.html)