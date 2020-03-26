# 前端技术专题

作为一个想自学一点前端知识的前端门外汉，先来对相关的东西做一个宏观概览。

首先，最基本的是　html, css 和　javascript. 这三部分是纯前端部分，后来出现的各种东西都是针对这三者在开发效率、代码维护、复用、构建等方面存在的问题而进行的补充／完善／优化。

例如，CSS 本身的设计有很多不利于工程化、影响开发效率的地方，这正是 CSS 预/后处理器要解决的问题：改进代码的组织和复用，更简洁地写嵌套，减少类名污染等。

typescript 是微软开源的‘更好的　javascript’（作者Anders Hejlsberg，C#之父），其代码经过编译成为标准　javascript 代码，并且它其实是　javascript 的超集，添加了严格的类型检查机制；其他如　Babel, CoffeScript 也是对　javascript 的扩展。

其次是一些模块化和流程化的工具。例如前端包管理器Bowe(和 npm 类似，只不过　npm 多用在后端而已).　require.js 可以实现 js 的异步加载，并管理依赖关系（但是 require.js 据说比较繁琐，用起来并不那么爽，已经式微）。类似的还有 Browserify.


最后是更为新潮的组件化框架。例如　React, Angular 2,  Polymer, Vue.

FLUX

参考　[2015前端组件化框架之路](http://www.cnblogs.com/aoldman/p/4445168.html)

### 模式之争

MVC(model-view-controller)
MVP(model-view-presenter)
MVVM(model-view-ViewModel)






---
### 多余的话

前端相对而言是一个比较新且知识更新非常快的领域，因此容易出现新老技术的对掐。
例如，经常会听到的鄙视链，React.js > AnjularJS > Jquery，2016年年初，更是爆发出尤小右和真阿当的互撕（参见庄表伟[《聊聊‘新技术’》](http://www.jianshu.com/p/e1425f186e3d?utm_campaign=haruki&utm_content=note&utm_medium=reader_share&utm_source=weibo)），堪称年度前端混战。我比较认同庄表伟的意见
>对于一个善于学习的人来说，走弯路他也在学习。对于不善于学习的人来说，就算在一条直道上，他也根本就不走啊！---何必担心年轻人被坑？担心他们走弯路？这样才有利于他们成长





