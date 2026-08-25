# emcee 默认 MCMC 采样机制

## 1. 采样目标

贝叶斯反演的目标是从后验分布中采样：

$$
p(\boldsymbol\theta\mid\boldsymbol y)
=
\frac{
p(\boldsymbol y\mid\boldsymbol\theta)
p(\boldsymbol\theta)
}{
p(\boldsymbol y)
}
$$

其中：

- $\boldsymbol\theta$：待反演参数；
- $\boldsymbol y$：实验观测数据；
- $p(\boldsymbol\theta)$：先验分布；
- $p(\boldsymbol y\mid\boldsymbol\theta)$：似然函数。

MCMC 不需要知道证据 $p(\boldsymbol y)$，因为这个常数会在接受率的比值中抵消。程序实际计算对数后验：

$$
\log p(\boldsymbol\theta\mid\boldsymbol y)
=
\log p(\boldsymbol\theta)
+
\log p(\boldsymbol y\mid\boldsymbol\theta)
+C
$$

---

## 2. Ensemble MCMC 与 walker

当前程序使用 `emcee.EnsembleSampler`，同时维护 $W$ 个 walker：

$$
\boldsymbol X_1,
\boldsymbol X_2,
\ldots,
\boldsymbol X_W
$$

每个 walker 的位置都是一套完整的参数。例如，四个反演参数可表示为：

$$
\boldsymbol X_k
=
\left[
\log D_{\mathrm{PyC}},
A_{\mathrm{PyC}},
\log D_{\mathrm{SiC}},
A_{\mathrm{SiC}}
\right]_k
$$

若：

```python
n_walkers = 12
dimension = 4
```

则初始状态的形状为：

```python
initial_state.shape == (12,4)
```

初始 walker 在用户给定的中心附近生成，且只保留落在先验支撑集内的候选点。

---

## 3. 默认游走方式：StretchMove

当创建 `EnsembleSampler` 时没有显式传入 `moves`，emcee 默认使用：

```python
emcee.moves.StretchMove(a=2.0)
```

考虑当前 walker $\boldsymbol X_k$，从其他 walker 中选择一个辅助 walker $\boldsymbol X_j$，再生成伸缩系数 $z$：

$$
z\in\left[\frac{1}{a},a\right]
$$

默认 $a=2$，因此：

$$
z\in[0.5,2]
$$

新的候选位置为：

$$
\boxed{
\boldsymbol X_k'
=
\boldsymbol X_j
+
z\left(
\boldsymbol X_k-\boldsymbol X_j
\right)
}
$$

这表示候选点位于两个 walker 的连线上：

- $z<1$：候选点在 $\boldsymbol X_j$ 与 $\boldsymbol X_k$ 之间；
- $z=1$：候选点与当前点相同；
- $z>1$：沿 $\boldsymbol X_j\rightarrow\boldsymbol X_k$ 方向继续向外伸展。

### 示例

只考虑两个参数：

$$
\boldsymbol\theta
=
[\log D_{\mathrm{SiC}},A_{\mathrm{SiC}}]
$$

设：

$$
\boldsymbol X_k=[-8.8,125000]
$$

$$
\boldsymbol X_j=[-9.0,120000]
$$

$$
z=1.5
$$

则：

$$
\begin{aligned}
\boldsymbol X_k'
&=
[-9.0,120000]
+1.5
\left(
[-8.8,125000]-[-9.0,120000]
\right)\\
&=[-8.7,127500]
\end{aligned}
$$

---

## 4. 候选参数在程序中的评估流程

每个候选位置 $\boldsymbol X_k'$ 都会转换成参数字典：

```python
theta = {
    "log_D_pyc": ...,
    "A_pyc": ...,
    "log_D_sic": ...,
    "A_sic": ...,
}
```

然后进入后验概率计算：

```text
theta
  ↓
计算 log_prior(theta)
  ↓
若超出先验范围，直接返回 -inf
  ↓
修改 MARS XML 输入
  ↓
运行 MARS
  ↓
获得预测释放曲线
  ↓
将预测结果插值到观测坐标
  ↓
计算残差、协方差与联合似然
  ↓
log_posterior = log_prior + log_likelihood
```

若候选点超出先验支撑集，程序会直接返回：

$$
\log p(\boldsymbol X_k'\mid\boldsymbol y)=-\infty
$$

此时不需要运行 MARS，且该候选点必然被拒绝。

---

## 5. StretchMove 的接受概率

对于 $D$ 维参数空间，候选点的接受概率为：

$$
\boxed{
\alpha
=
\min\left[
1,
z^{D-1}
\frac{
p(\boldsymbol X_k'\mid\boldsymbol y)
}{
p(\boldsymbol X_k\mid\boldsymbol y)
}
\right]
}
$$

程序在对数空间中等价地判断：

$$
\log u
<
(D-1)\log z
+
\log p(\boldsymbol X_k'\mid\boldsymbol y)
-
\log p(\boldsymbol X_k\mid\boldsymbol y)
$$

其中：

$$
u\sim U(0,1)
$$

- 若接受：

$$
\boldsymbol X_k^{(t+1)}=\boldsymbol X_k'
$$

- 若拒绝：

$$
\boldsymbol X_k^{(t+1)}=\boldsymbol X_k^{(t)}
$$

因此，MCMC 链中连续出现相同样本是正常现象。

即使候选点的后验概率较低，也仍然可能被接受。这使算法能够探索完整后验分布，而不是只向最大后验点爬升。

### $z^{D-1}$ 的作用

StretchMove 的候选分布不是普通的对称高斯随机游走。$z^{D-1}$ 是对正向和反向提议概率不对称的修正，用于保证详细平衡，使采样链的稳定分布确实是目标后验分布。

---

## 6. Red–Blue 分组更新

每个 walker 的新位置都依赖其他 walker。为了在满足详细平衡的同时并行评估，emcee 将 walker 随机分为两个子集：

```text
Red 组
Blue 组
```

一次完整更新为：

```text
1. 固定 Blue 组
2. 使用 Blue 中的 walker 为 Red 生成候选点
3. 并行计算所有 Red 候选点的后验
4. 分别接受或拒绝 Red 的候选点
5. 固定更新后的 Red 组
6. 使用 Red 中的 walker 更新 Blue
```

完成 Red 和 Blue 两个子集的更新后，才算完成一个 MCMC step。

---

## 7. 多进程并行的位置

多进程不改变 MCMC 的数学算法，只并行执行同一子集内多个候选点的后验计算。

例如，12 个 walker 被分为两组，每组 6 个。更新 Red 时会生成 6 个候选点：

```text
theta'_0 → 工作进程1 → MARS → log posterior
theta'_1 → 工作进程2 → MARS → log posterior
theta'_2 → 工作进程3 → MARS → log posterior
...
```

候选点后验计算完成后，主采样器再根据接受准则更新各个 walker。

因此，多进程只改变计算速度，不改变：

- StretchMove 的候选点公式；
- Metropolis–Hastings 接受概率；
- 目标后验分布；
- MCMC 链的数学含义。

---

## 8. 一个完整 MCMC step 的流程

```text
当前存在 W 个 walker
        ↓
随机分为 Red 与 Blue
        ↓
固定 Blue
        ↓
为每个 Red walker：
    选择一个 Blue walker
    生成伸缩系数 z
    构造候选参数 theta'
        ↓
并行计算所有 Red 候选点：
    检查先验
    运行 MARS
    计算似然
    返回 log posterior
        ↓
对每个 Red walker 接受或拒绝
        ↓
固定更新后的 Red
        ↓
使用相同方法更新 Blue
        ↓
得到下一个 ensemble 状态
```

---

## 9. Burn-in 与正式采样

Burn-in 与正式采样使用完全相同的 StretchMove 和接受准则。两者的区别只在于结果是否进入后验统计：

```text
Burn-in：
    StretchMove → 接受/拒绝 → 不保存到正式链

Production：
    StretchMove → 接受/拒绝 → 保存并用于后验统计
```

Burn-in 使 walker 从人为指定的初始位置移向后验高概率区域；正式采样阶段才用于估计后验分布、置信区间与参数相关性。

---

## 10. StretchMove 的特点

StretchMove 具有仿射不变性，对参数空间的线性缩放和旋转不敏感。因此，它比简单的各向同性高斯随机游走更适合处理：

- 参数单位不同；
- 参数尺度差异很大；
- 后验分布呈倾斜椭圆形；
- 参数之间存在近似线性相关。

但它仍可能在以下情况中混合缓慢：

- 后验分布具有多个相距很远的模态；
- 所有 walker 初始时都落入同一个局部模态；
- 后验存在强烈非线性弯曲；
- 参数维数较高。

因此，采样完成后仍需检查：

- walker 轨迹；
- 接受率；
- 积分自相关时间 $\tau$；
- 有效样本量 $N_{\mathrm{eff}}$；
- 链长是否足以支持稳定的后验统计。

