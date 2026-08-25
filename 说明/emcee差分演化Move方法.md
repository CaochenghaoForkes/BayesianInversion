# emcee 差分演化 Move 方法

## DEMove

`DEMove` 是差分演化提议。更新某个 walker \(\boldsymbol\theta\) 时，从互补 walker 集合中选择两个位置 \(\boldsymbol\theta_a\) 和 \(\boldsymbol\theta_b\)，利用二者的差向量构造候选位置：

$$
\boldsymbol\theta'
=
\boldsymbol\theta
+
\gamma\left(\boldsymbol\theta_a-\boldsymbol\theta_b\right)
+
\boldsymbol\epsilon
$$

其中：

- \(\boldsymbol\theta_a-\boldsymbol\theta_b\) 反映当前后验样本云的尺度和相关方向；
- \(\gamma\) 控制跳跃幅度；
- \(\boldsymbol\epsilon\) 是很小的随机扰动，避免提议退化。

emcee 默认根据参数维数设置 \(\gamma\) 的典型尺度。因此，它通常不需要像普通高斯随机游走那样，手动为每个参数设计合适的提议标准差。

对于扩散系数和活化能反演，如果两个参数在后验中形成倾斜的相关带，walker 之间的差向量往往也沿着这条相关带。`DEMove` 因而可以顺着后验的主要方向移动，通常比各方向独立扰动更有效。

### 与普通随机游走的直观比较

以扩散指前因子和活化能为例：

$$
D(T)=D_0\exp\left(-\frac{A}{RT}\right)
$$

提高 \(D_0\) 的同时提高 \(A\)，可能仍然得到相近的有效扩散系数。因此，\(\log D_0\) 与 \(A\) 的后验高概率区域可能呈倾斜的狭长带，而不是横平竖直的圆形区域：

```text
A
│                    •
│                •
│            •
│        •
│    •
└──────────────────────── logD
```

如果普通高斯随机游走只对两个参数分别施加独立扰动，它提出的候选点容易沿坐标轴方向横着或竖着移动，从而离开这条高概率带，导致较多候选点被拒绝。

普通随机游走并非必然只能横着或竖着跳。只要预先给它设置合适的非对角协方差矩阵，它同样可以沿倾斜方向移动；困难在于这个协方差和参数尺度通常需要预先估计并调节。

`DEMove` 不需要预先指定这条倾斜方向。假设两个参考 walker 位于：

$$
\boldsymbol\theta_a=(-3.00,230000),
\qquad
\boldsymbol\theta_b=(-3.40,210000)
$$

它们的差向量为：

$$
\boldsymbol\theta_a-\boldsymbol\theta_b
=(0.40,20000)
$$

这个差向量同时包含：

- 各参数在当前后验区域中的典型变化尺度；
- 两个参数共同增减形成的相关方向。

如果 walker 已经沿倾斜的高概率带分布，那么随机选择的两个 walker 之间的差向量也很可能沿着这条带。`DEMove` 用该差向量推动当前 walker，因此候选点更可能继续落在高概率区域附近：

$$
\boxed{
\text{候选位置}
=
\text{当前位置}
+
\gamma\times\text{两个参考 walker 的差}
}
$$

所以，`DEMove` 的关键并不是提前知道扩散参数之间存在何种补偿关系，而是从当前 ensemble 的几何分布中动态获得参数尺度和相关方向。候选点最终是否被采用，仍然由先验和 MARS 似然共同决定。

### DEMove 的特点

- 适合存在明显参数相关性的后验；
- 自动利用当前 ensemble 的尺度和方向；
- 对不同参数量纲的适应能力较好；
- 如果所有 walker 都集中在同一个局部区域，它本身不一定能够轻易发现距离很远的另一个模态。

## DESnookerMove

`DESnookerMove` 同样利用 walker 之间的差异，但它不像 `DEMove` 那样直接沿两个参考 walker 的差向量移动。它使用三个参考 walker：

- \(\boldsymbol z\)：锚点，用来确定当前 walker 的移动方向；
- \(\boldsymbol z_1,\boldsymbol z_2\)：二者的差用来决定移动距离；
- \(\boldsymbol\theta\)：当前准备更新的 walker。

### 确定移动方向

从锚点 \(\boldsymbol z\) 指向当前 walker：

$$
\boldsymbol d
=
\boldsymbol\theta-\boldsymbol z
$$

其单位方向为：

$$
\boldsymbol u
=
\frac{\boldsymbol\theta-\boldsymbol z}
{\lVert\boldsymbol\theta-\boldsymbol z\rVert}
$$

当前 walker 将沿着锚点与当前位置确定的直线移动：

```text
z ───────────── θ ───────────── θ'
锚点          当前位置           候选位置
```

这类似台球沿球杆瞄准方向运动，也是 `snooker` 名称的直观来源。

### 确定移动距离

计算另外两个参考 walker 的差：

$$
\boldsymbol z_1-\boldsymbol z_2
$$

Snooker Move 不直接沿该差向量移动，而是取它在 \(\boldsymbol u\) 方向上的投影：

$$
\Delta
=
\boldsymbol u^{\mathsf T}
(\boldsymbol z_1-\boldsymbol z_2)
$$

候选点可以概念性地写成：

$$
\boxed{
\boldsymbol\theta'
=
\boldsymbol\theta
+
\gamma_s\Delta\boldsymbol u
}
$$

emcee 默认使用 \(\gamma_s=1.7\)。\(\Delta\) 的正负决定沿该直线向哪个方向移动，其绝对值决定移动距离。

### 与 DEMove 的核心区别

$$
\boxed{
\text{DEMove：两个参考 walker 的差同时决定方向和距离}
}
$$

$$
\boxed{
\text{DESnookerMove：锚点与当前 walker 决定方向，}
\\
\text{另两个参考 walker 的差决定距离}
}
$$

`DEMove` 往往沿当前样本云已有的主要差分方向高效移动。`DESnookerMove` 每次随机改变锚点，由此产生不同的几何方向，可以偶尔提供方向更特殊、跨度更大的跳跃。

它的主要作用不是替代 `DEMove`，而是补充 `DEMove`：

- 改变 walker 的常规移动方向；
- 帮助采样器离开局部狭窄区域；
- 对轻度多峰、弯曲或几何结构较复杂的后验可能更有效；
- 单独大量使用时未必具有最高的局部采样效率。

如果所有 walker 都集中在同一个局部区域，`DESnookerMove` 同样不能凭空获得远处模态的信息。

### 几何接受率修正

Snooker Move 沿锚点射线伸缩时，从当前位置跳到候选位置和反向跳回的几何体积并不完全相同。因此，接受概率除了比较新旧位置的后验概率，还需要包含与下式有关的几何修正：

$$
\frac{
\lVert\boldsymbol\theta'-\boldsymbol z\rVert
}{
\lVert\boldsymbol\theta-\boldsymbol z\rVert
}
$$

该修正确保提议满足 MCMC 所需的平衡条件，emcee 已在 `DESnookerMove` 内部自动完成，程序不需要额外计算。

## 参考资料

- emcee Moves 官方文档：<https://emcee.readthedocs.io/en/stable/user/moves/>
- emcee 混合 Move 示例：<https://emcee.readthedocs.io/en/stable/tutorials/moves/>
- Ter Braak, C. J. F. and Vrugt, J. A. (2008), Differential Evolution Markov Chain with snooker updater and fewer chains.
