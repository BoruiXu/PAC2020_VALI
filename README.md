# PAC2020_VALI
this is my code for PAC2020


很高兴参与了PAC2020的比赛，虽然中间有很多波折，最后也没有进入决赛，不过还是有不少感悟的，因为是第一次真正去做优化的东西，因此记录一下初赛题目的优化思路和心得。

首先是赛题回顾：
 
初赛题目：
傅里叶空间图像相似度计算
□ 赛题简介：
在冷冻电镜三维重构程序中，将二维真实图像与空间中的三维结构的投影图像的相似度计算是调用最为频繁的计算，相似度计算的原理是真实图像与投影图像的所有像素在傅里叶空间中的二范数之和，公式如下：
![image](https://github.com/a919480698/PAC2020_VALI/tree/master/picture/1.png)
□ 赛题要求 ：
1. 解压源码包后，根目录下有以下文件：
a）main.cpp：计算的主程序。
b）nput.dat：输入数据文件；K.dat：为减小输入文件体积而增加的扰动因子。不可修改！
c）check.dat：标准结果输出文件，可用于验证程序的计算结果，每个数据的有效数字允许不大于十万分之一的误差。不可修改！
d）Makefile：参赛队员将使用的编译器参数写入Makefile。
2. 比赛考察程序计时部分的运行时间，从数据文件读入结束开始，到结果文件输出结束终止，时间戳的位置不可修改！
3. Main.cpp中数据文件读取部分不包括在程序计时内。
4. Main.cpp中的m和K值。不可修改！
5. Main.cpp 中函数logDataVSPrior 的最后一行代码return result*disturb0; 不可修改！
6. 参赛队员需要手动写出向量化SIMD优化代码。
7. 可以改变数据结构或者数据类型。
8. 如认为有必要，可以将必要的代码修改为其它语言如Fortran、汇编、intrinsic等等。
9. 为控制输入文件的体积，K次迭代都使用相同的数据文件，并加入一个扰动因子模拟每次迭代处理不同的数据，所以数据计算必须放在logDataVSPrior函数中，logDataVSPrior函数必须调用K次。利用每次迭代使用同一组数据而减少计算次数的取消成绩！


以上是题目的总体要求，在main函数中主要计时部分如下图所示：

![image](https://github.com/a919480698/PAC2020_VALI/tree/master/picture/12.png)

从上图可以看到主要耗时为logDataVSPrior函数部分，该部分一共重复了K次。logDataVSPrior函数如下图所示：


![image](https://github.com/a919480698/PAC2020_VALI/tree/master/picture/3.png)
logDataVSPrior函数主要由一个for循环和累加计算构成，十分简单。


瓶颈分析：

在拿到题目后首先需要对数据量以及理论瓶颈进行分析，从而对以后的实际优化过程起到指导作用。
在该题目中，进行执行一次logDataVSPrior函数计算即为对图像进行一次相似度的计算，其中图像的大小（长度）m=1638400，共执行了K次相似度计算，K=100000次。

首先对数据量进行分析，针对于每一次logDataVSPrior函数的执行，一共需要执行循环m次，每一次循环需要6个double类型的数据（complex是两个double），则执行完一次logDataVSPrior函数需要的数据量为1638400*6=9830400个double类型的数据，再考虑K次循环以及一个double是8 Byte，所以总的数据量是9830400*8*100000 = 7.86T，所以完成所有计算，总的数据量大概是7.86T。

接下来针对于如下机器对访存瓶颈和计算瓶颈进行分析，机器配置：
cpu max 3.9GHz  8 core
Ram 16G 2666MHz  2 channels 

对于访存来说，每秒最多数据读写量为2666*8*2 = 0.042656 T/s，所以针对于7.86T的数据，如果需要读入这些数据量的数据，那么完成该部分代码最短用时大于等于7.86/0.042656，约等于184s。

对于CPU来说，针对于double类型数据，计算峰值性能为：CPU核数*CPU频率*每个周期执行的浮点操作数。针对于avx512指令集，每次可以同时计算8个douuble类型的数据，考虑FMA指令集可以加法和乘法同时计算，CPI：每条指令执行所需要的周期，对于Intel CPU，每个周期可以执行两个FMA指令。所以峰值性能为3.9G*8*2*2*8 = 998.4GFlops，所以完成7.86T数据计算最少需要大约7.87s（实际使用avx512会降频）

而在给出的代码中，未进行任何优化，运行时间为8987s，约为2.5小时。可见既没有到达访存瓶颈，也没有到达计算瓶颈。

代码优化：

首先考虑访存方面的问题，在目前阶段，可以注意到在循环的K次中，K次数据是完全相同的，然而由于m太大，因此在第二次K循环的时候，本已经在缓存中的数据被替换到了RAM中，这大大增加了访存的成本，因此如果能够在K次循环中，重复利用到数据的空间局部性，则会大大提高访存效率。因此这里可以考虑使用cache blocking策略。对于长度为m的图像数据，先计算数据的前j项，对前j项数据先循环K次，再对（j，2j）的数据进行K次循环，以此类推…………从而提高数据的访存效率

之后便是考虑多线程的并行问题，这里使用OpenMP多线程处理，因为m的每一个数据块需要循环K次，因此使用多线程并行处理K次的for循环。

对于logDataVSPrior函数内部是采用了手动SIMD的处理方法，利用intrinsic实现SIMD。使用avx512指令集以及FMA指令集，并且因为logDataVSPrior函数中循环之间存在依赖性，这里将循环展开，每次循环处理16组数据，最后再累加存储结果。

其他方面，原程序中使用了fout将结果输出到文件，每次输出使用endl结尾，这样会每次输出都会刷新缓冲区，导致频繁访问IO，可以将endl更换为“\n”减少IO的访问次数，最后flush刷新缓冲区。

还需要注意的一点是，以上的优化思路均是在单CPU情况下考虑的，为考虑NUMA节点的情况。如果机器含有NUMA节点，那么在访存过程中还会存在跨numa节点访存的情况，这也会大大影响访存的效率。这里采取的方法是MPI+OpenMP混合编程的方法。针对于每个numa节点创建一个进程，这样每个进程上（即每个numa节点）都会读取到原始的输入数据，不需要跨节点访问输入数据，每个节点再利用多线程处理K次循环的一部分，最后使用MPI_Gather汇集数据统一输出到文件中。

以上便是对于PAC2020初赛题目的整体优化思路，如存在问题，未来会及时修改订正。
