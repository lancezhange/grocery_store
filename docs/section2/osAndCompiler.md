# 1. 操作系统和编译原理

编译步骤

1. 词法分析 有限状态自动机
2. 语法分析 栈的下推自动机，产生 Parse Tree.
3. 代码生成
4.

## 1.1. 编译器的分支预测

例如如下**循环**中的 if 条件语句
if(data[i] >= SOME_VALUE){
//do_something
}
如果 data 本身是排好序的，其用时会显著少于 data 无序的情形，这就是因为编译器会做分支预测，当 data 有序时，不会频繁跳转分支。

## 1.2. 非规格化浮点数(Denormal Number)

非规格化浮点数的处理比较耗时

## 1.3. 页面调度算法

包括 LRU, LFU, FIFO 等，注意区别 LRU 和 LFU，前者是从频率上，后者是从时间上。

## 1.4. Linux 内存管理

## 1.5. 虚拟内存地址与物理内存地址

### 1.5.1. 资料

[gitbook: linux insides](https://www.gitbook.com/book/0xax/linux-insides/details)
[gitbook: 理解 linux 进程](http://www.linuxprocess.com/)
