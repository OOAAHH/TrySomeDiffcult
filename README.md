# TrySomeDiffcult

**目录**
[第一次尝试](https://github.com/OOAAHH/TrySomeDiffcult/tree/main?tab=readme-ov-file#2024119)
[第二次尝试](https://github.com/OOAAHH/TrySomeDiffcult?tab=readme-ov-file#2024830)

## 2024.1.19
  我的NW算法的实现，这个东西蛮有意思的。@jit/@njit来做加速函数的部分，发现整体上来看反而不快，比较奇怪。

  在代码层面，numba装饰的函数在循环体中进行运转反而不快，要优化还是得看并行化编程，或者我直接用C写一遍，抑或着是信息办法改掉数据结构，让循环体也能被@jit修饰。在算法层面，我应该尝试优化算法结构和数据结构，我使用了python原生的list对象，这导致我在进行numba加速的时候被拖累了。
<img width="893" alt="截屏2024-01-19 11 30 28" src="https://github.com/OOAAHH/TrySomeDiffcult/assets/19518905/20e5747a-b0c2-47d0-9da9-a402c039a0c9">


----------------------
## 2024.8.30
实验室新人来做NW算法了，我来尝试在上次的基础上用C语言重写一遍。上次的代码就其功能而言是完备的，也能处理复杂序列的情况，在不考虑时间消耗的情况下。
但是，当我们必须考虑时间及产生的序列的可用性的时候，之前的代码就不合适了。

我首先想到的就是把python代码转换为C的代码，这部分的问题依旧出在性能上。我使用的用来做测试的序列如下：

```C
// 两个需要比对的DNA序列
char A[] = "ATCGGGCTACATCGGGCTACGGATCGGGCTACGAAAAAAAAAAAAAAAAAAAAAA";
char B[] = "ATCGGATCGGGCTACGATCGGGCTACGATCGCGTTTTTTTAAAAAGAAAAAAAAGGGGGGGGTGTATTGTA";
```

在实践中我发现，如果我继续用遍历所有可能路径的方式来做这个序列的比对，其可能的路径的数量级在“亿”的级别。
即使用C语言来编写搜索算法也很慢，我开始寻求并行化的方法。现在的做法是我们依次对每个可能的路径依次搜索，作为并行化的更改我们在这一步进行拆分，并行的进行每个路径的遍历。

在并行化这一步遇到了新的问题，并行化的时候遇到了`Segmentation fault (core dumped)`的问题。看起来是在内存分配上遇到了问题。在这一步，我选择了“剪枝”的思路来处理这个问题。即考虑到比对结果的可用性，在比对的前期对可能的路径进行选择，去除一部分结果。
然而，这样的思路导致了另外一个问题，一些在前期表现较好的比对结果更容易出现在最终的结果中，而那些在稍微靠后的位置上才表现优异的比对结果就无法体现了。

~~我本来就是实现贪婪算法，再“贪婪”一点也不是问题~~

这个问题的本质是：
    - 早期剪枝：可能会提前剪掉一些后续整体得分较高的路径，从而导致最终结果不佳。
    - 局部最优：在路径生成过程中，可能会优先保留局部最优的路径，而忽略了全局最优的路径。

为了避免这个问题，可以考虑以下几种策略：

1. 延迟剪枝：
先生成所有可能的路径，计算它们的得分，然后在所有路径中选择得分最高的前5000个。这种方法避免了提前剪枝带来的问题，但代价是需要更多的内存和计算时间。

2. 分段剪枝：
在每个部分（即从矩阵的某一行到某一行，或某一列到某一列）进行剪枝，而不是在每个步骤进行剪枝。这样可以让算法有机会探索更多的路径组合，从而增加全局最优解出现的可能性。

3. 比如优先队列剪枝（Best-First Search）：
使用优先队列存储路径，路径的优先级基于当前的总得分（已计算部分的得分 + 对未计算部分的一个估计得分）。每次扩展时，从队列中取出优先级最高的路径进行扩展，这样可以尽量保留潜在得分高的路径。
**但是这样做也有显著的缺点，那就是这样的结果并不一定是最优的**

说到这里，就不得不提路径搜索的三大 baseline 算法 Dijkstra（荷兰语 迪斯科特拉）、Best-first 以及 A* search 算法。
今天不进行展开，只先尝试用best-first先完成这个算法作业。可能对于人类而言，只需要前10个比对结果就OK👌了。
代码附在文件`NW实现3.c`中。

![图片](https://github.com/user-attachments/assets/2b52d4e2-74dd-4787-980c-8ee85b9ba6e8)

