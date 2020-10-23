#  Related Works Review

## Factorization Machines https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

关键的两个公式

模型方程

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+y+%3D%7B%7D+%26+w_0%2B%5Csum_%7Bi%3D1%7D%5Enw_ix_i%2B%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%5Csum_%7Bj%3Di%2B1%7D%5En%5Clangle+V_i%2CV_j%5Crangle+x_ix_j+%5Cnotag+%5C%5C+%3D%7B%7D+%26+w_0%2B%5Csum_%7Bi%3D1%7D%5Enw_ix_i%2B%5Cfrac%7B1%7D%7B2%7D%2A%5Csum_%7Bf%3D1%7D%5E%7Bk%7D%5Cleft%5C%7B%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dv_%7Bif%7Dx_i%5Cright%29%5E%7B2%7D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dv_%7Bif%7D%5E%7B2%7Dx_%7Bi%7D%5E2%5Cright%5C%7D+%5Cnotag+%5C%5C+%5Cend%7Balign%7D+%5Ctag%7B4%7D)

方法各参数的梯度表达式

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cfrac%7B%5Cpartial%7By%7D%7D%7B%5Cpartial%7B%5Ctheta%7D%7D+%3D++%5Cbegin%7Bcases%7D+1%2C+%26+%5Ctext%7Bif+%7D+%5Ctheta+%5Ctext%7B+is+%7D+w_0%3B+%5C%5C+x_i%2C+%26+%5Ctext%7Bif+%7D+%5Ctheta+%5Ctext%7B+is+%7D+w_i%3B+%5C%5C+x_i%5Csum_%7Bj%3D1%7D%5E%7Bn%7Dv_%7Bjf%7Dx_j-x_%7Bi%7D%5E2v_%7Bif%7D%2C+%26+%5Ctext%7Bif+%7D+%5Ctheta+%5Ctext%7B+is+%7D+v_%7Bif%7D.+%5Cend%7Bcases%7D+%5Cend%7Bequation%7D+%5Cnotag)

手动实现模型的话，主要的工作就是要实现第一个方程，以及每次更新W0,Wi,Vi,f。做好这两个这个模型就基本成功实现。

为了克服上述困难，需要对FM公式进行改写，使得求解更加顺利。受 **矩阵分解** 的启发，对于每一个特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i)  引入辅助向量（隐向量） ![[公式]](https://www.zhihu.com/equation?tex=V_i%3D%28v_%7Bi1%7D%2Cv_%7Bi2%7D%2C%5Ccdots%2Cv_%7Bik%7D%29) ，然后利用 ![[公式]](https://www.zhihu.com/equation?tex=V_iV_j%5ET) 对 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D) 进行求解。即，做如下假设： ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D+%5Capprox+V_iV_j%5ET)  。

引入隐向量的好处是：

1. 二阶项的参数量由原来的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7Bn%28n-1%29%7D%7B2%7D)  降为 ![[公式]](https://www.zhihu.com/equation?tex=kn)  。
2. 原先参数之间并无关联关系，但是现在通过隐向量可以建立关系。如，之前 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D)  与  ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bik%7D) 无关，但是现在 ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bij%7D%3D%5Clangle+V_i%2CV_j%5Crangle+%2Cw_%7Bik%7D%3D%5Clangle+V_i%2CV_k%5Crangle+)  两者有共同的  ![[公式]](https://www.zhihu.com/equation?tex=V_i) ，也就是说，所有包含  ![[公式]](https://www.zhihu.com/equation?tex=x_ix_j) 的非零组合特征（存在某个 ![[公式]](https://www.zhihu.com/equation?tex=j%5Cneq+i)  ，使得 ![[公式]](https://www.zhihu.com/equation?tex=x_ix_j%5Cneq+0)  ）的样本都可以用来学习隐向量 ![[公式]](https://www.zhihu.com/equation?tex=V_i)  ，这很大程度上避免了数据稀疏性造成的影响。

### **优缺点**

优点：

1. 它可以自动学习两个特征间的关系，可以减少一部分的交叉特征选择工作，而且参数也不算多，调起来不会太痛苦。

2. 因为不需要输入那么多的交叉特征，所以产生的模型相对LR的模型会小很多。

3. 在线计算时减小了交叉特征的拼装，在线计算的速度基本和LR持平（虽然两个向量的点积的计算看上去会导致计算量翻了几倍）。

缺点：

1. 每个特征只引入了一个隐向量，不同类型特征之间交叉没有区分性。FFM模型正是以这一点作为切入进行改进。

## librec中的FM实现(FMSGD)

主要是看了FMSGDRecommender、FactorizationMachineRecommender、TensorRecommender这三个类，以及当中用到的自定义数据结构，比如math.structure里的vector大类的结构。

一开始看的是FactorizationMachineRecommender，苦于没有找到实现train的代码，停滞了很久，后来 才发现FactorizationMachineRecommender是个抽象类，终于找到了我想找的FMSGDRecommender，损失函数是随机梯度下降。

主要的疑点在于下列三个公式的实现

<img src="C:\Users\shw76\AppData\Roaming\Typora\typora-user-images\image-20201022150641455.png" alt="image-20201022150641455" style="zoom:50%;" />

```java
for (VectorEntry ve2 : vector) {
    int j = ve2.index();
    if (j != i) {
        hVif += xi * V.get(j, f) * ve2.get();
    }
}
```

为啥公式里还有一个Vi,f*(Xi^2)，这里没有。同时if(j!=l)的判断是哪里的。

答：观察公式可得，在求![image-20201022151538650](C:\Users\shw76\AppData\Roaming\Typora\typora-user-images\image-20201022151538650.png)

的时候就已经求了Vi,f(Xi^2)一次，外面的一项只是减去这个。那么改写成我们的代码的话其实就可以直接在for循环1~n里面，用一个if判断此时是否i==j，如果等于跳过这一次。

> 不过这里for循环判断的话，时间复杂度是O(n)，而直接计算完了之后减去的话时间复杂度是O(1)。这里可能还真有优化空间。
>
> https://github.com/guoguibing/librec/pull/339 提交了一个pr。

## 推荐系统召回四模型之：全能的FM模型https://zhuanlan.zhihu.com/p/58160982

看完这篇的感受是工业应用的模型要贴合数据本身，证明LR等建议模型在公司中继续应用的价值，毕竟简易、计算量小。

企业内用LR、DNN做推荐，横向比较起来，LR容易出效果，费心调DNN参数吃力不讨好。

FM模型也直接引入任意两个特征的二阶特征组合，和SVM模型最大的不同，在于特征组合权重的计算方法。FM对于每个特征，学习一个大小为k的一维向量，于是，两个特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=x_j) 的特征组合的权重值，通过特征对应的向量 ![[公式]](https://www.zhihu.com/equation?tex=v_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=v_j) 的内积 ![[公式]](https://www.zhihu.com/equation?tex=%E2%8C%A9v_i%2Cv_j+%E2%8C%AA) 来表示。这本质上是在对特征进行embedding化表征，和目前非常常见的各种实体embedding本质思想是一脉相承的

> 万物皆可embedding，embedding算法选择的好，直接影响模型的训练效果。

MF（Matrix Factorization，矩阵分解）模型是个在推荐系统领域里资格很深的老前辈协同过滤模型了。核心思想是通过两个低维小矩阵（一个代表用户embedding矩阵，一个代表物品embedding矩阵）的乘积计算，来模拟真实用户点击或评分产生的大的协同信息稀疏矩阵，本质上是编码了用户和物品协同信息的降维模型。

本质上，MF模型是FM模型的特例，MF可以被认为是只有User ID 和Item ID这两个特征Fields的FM模型，MF将这两类特征通过矩阵分解，来达到将这两类特征embedding化表达的目的。而FM则可以看作是MF模型的进一步拓展，除了User ID和Item ID这两类特征外，很多其它类型的特征，都可以进一步融入FM模型里，它将所有这些特征转化为embedding低维向量表达，并计算任意两个特征embedding的内积，就是特征组合的权重，如果FM只使用User ID 和Item ID，你套到FM公式里，看看它的预测过程和MF的预测过程一样吗

![img](https://pic3.zhimg.com/80/v2-99c833582ffc55a3c108ca4096321542_1440w.jpg)

> 在企业里，效率和指标都需要考虑，显然FM模型公式在未化简前是O(k*n^2)的时间复杂度，优化后O(k*n)
>
> 同时Link里还有FM模型公式的推导

# [深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)

假设样本的 nn 个特征属于 ff 个field，那么FFM的二次项有 nfnf个隐向量。而在FM模型中，每一维特征的隐向量只有一个。FM可以看作FFM的特例，是把所有特征都归属到一个field时的FFM模型。根据FFM的field敏感特性，可以导出其模型方程。

![image-20201022210540950](C:\Users\shw76\AppData\Roaming\Typora\typora-user-images\image-20201022210540950.png)

其中，fjfj 是第 jj 个特征所属的field。如果隐向量的长度为 kk，那么FFM的二次参数有 nfknfk 个，远多于FM模型的 nknk 个。此外，由于隐向量与field相关，FFM二次项并不能够化简，其预测复杂度是 O(kn2)O(kn2)。

![image-20201022211113026](C:\Users\shw76\AppData\Roaming\Typora\typora-user-images\image-20201022211113026.png)

> 这张图解释了之前librec源码里面为啥是
>
> ![image-20201022211409152](C:\Users\shw76\Desktop\Algorithm\大数据笔记\image-20201022211409152.png)
>
> ----------------------------------------------------------------------------------------------------------------------------------------------------
>
> ![image-20201022211829336](C:\Users\shw76\AppData\Roaming\Typora\typora-user-images\image-20201022211829336.png)
>
> 当one-hot化之后，很多特征都需要拆分，比如日期、地点这种，因为它们都有多个取值