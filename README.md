### Back-Propagation Multi Dimensional Neural Network
#### Author: Ying Yan, Yun Tang, Guanting Liu

BP-MTN 基于传统的多维泰勒网络（MTN），通过添加全连接层、softmax 层和 ReLU 激活函数等改进，使其能够处理故障诊断中的分类问题。具体而言，全连接层解决了故障特征维度与类别数量不匹配的问题，softmax 层实现了分类功能，ReLU 函数提高了分类准确性并降低了模型复杂度。在训练方面，使用基于小批量梯度下降算法的 BP 算法来训练 BP-MTN 分类器，而不是传统 MTN 中使用的非线性最小二乘法。本文将其应用于HVAC AHU的故障诊断。通过实验，对多项式阶数和激活函数的选择进行了探索，发现 1 阶多项式的模型具有较高性价比，当使用 1 阶多项式时，ReLU 函数表现最佳，当使用 2 阶或 3 阶多项式时，Leaky ReLU 函数表现更好。实验结果表明，BP-MTN 方法能有效实现 AHU 故障的准确分类，且相比其他一些现有方法表现更优。

MTN - 主函数

Activate - 激活函数

Activate_grad - 激活函数导数

Gradient_renewal - 梯度更新方法

Initialization - 初始化方法

Normalization - 归一化方法

Taylor_expan - 多项式层实现

feature_selected - 示例数据

需先安装“Parallel Computing Toolbox”和“Statistics and Machine Learning Toolbox”工具箱

#### 使用该方法后需引用以下论文：
Yan, Y., Cai, J., Tang, Y., & Chen, L. (2023). Fault diagnosis of HVAC AHUs based on a BP-MTN classifier. Building and Environment, 227, 109779.
