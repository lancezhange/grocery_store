# Java {ignore=true}

[TOC]

## 硬语法

静态方法可以用类名调用，即调用静态方法可以无需创建对象，这意味着静态方法中只能访问类的静态成员和静态方法，而不能访问一般的其他（属于实例的）变量和方法。正因如此，静态方法中不允许使用 `this`.

不像 scala, java 中除了引用类型，还有基本类型（又称**原生类型**）

### Java 8 特性

Java 8 引入了 lambda 表达式、接口中的默认方法（对应于 scala 中的 traits）、集合(collections)上的流操作。stream 很怪异啊。

### Java 9 特性

- jshell/project Kulla 这是　 Java 首个官方　 REPL
- Java Microbenchmarking Harness (JMH)
- 全面支持 HTTP 2,0

## 高质量库

- [joda-time](http://www.joda.org/joda-time/)

## JVM

### 类的加载

JVM 运行时动态加载和动态连接

JIT(just in time)
即时编译，加快执行速度

我们知道，Java 代码先要被 javac 编译成字节码，然后被 jvm 解释翻译为机器指令来执行，这就比可执行的二进制字节码慢了一些。通过 JIT 技术，可以提高代码的执行速度，这是怎么做到的呢？

## 参考资料

- [Java 资源大全中文版](https://github.com/jobbole/awesome-java-cn)　一份详尽的　 Java 资源列表

- [Thinging in Java]()
- [Effective Java]()
- [Java Puzzlers: Traps, Pitfalls, and Corner Cases]()

  作者为　 Joshua Bloch 与　 Neal Gafter. 'We reserve the right to mislead you'.

- [Java 工程师成神之路](http://www.importnew.com/17389.html)
- [更好的 java](http://www.importnew.com/16160.html)
- [Aleksey Shipilëv: One Stop Page](http://shipilev.net/)
- [Java8 - The Missing Tutorial](https://github.com/shekhargulati/java8-the-missing-tutorial)
