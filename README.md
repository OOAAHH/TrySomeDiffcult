# TrySomeDiffcult

- 2024.1.19
  我的NW算法的实现，这个东西蛮有意思的。@jit/@njit来做加速函数的部分，发现整体上来看反而不快，比较奇怪。

  在代码层面，numba装饰的函数在循环体中进行运转反而不快，要优化还是得看并行化编程，或者我直接用C写一遍，抑或着是信息办法改掉数据结构，让循环体也能被@jit修饰。在算法层面，我应该尝试优化算法结构和数据结构，我使用了python原生的list对象，这导致我在进行numba加速的时候被拖累了。
<img width="893" alt="截屏2024-01-19 11 30 28" src="https://github.com/OOAAHH/TrySomeDiffcult/assets/19518905/20e5747a-b0c2-47d0-9da9-a402c039a0c9">
