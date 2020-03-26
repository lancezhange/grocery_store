# 日常工具

### 编辑器相关

- sublime text2/3

  就算不是下雨天，sublime 和如下插件更配哦

  - package control 包管理，首要插件
    这里讲一下我之前遇到的 在 mac 下安装时遇到的坑。手动将下载的 package control 安装包放到 sublime 安装路径对应的 `Installed Packages` 目录下，但是重启之后怎么都不生效。
    后来才知道原因竟然是配置文件竟然配置了将 package control 包忽略！将对应的配置去掉就好了。
  - keymaps 展现你所有的 keymap, 方便管理和查询快捷键，真心牛逼
  - sidebar enhangcement 侧边栏增强
  - advanced newfile `ctrl alt n` 创建文件，tab 键可以补全哦
  - auto filename 自动补全文件名
  - SFTP 本地和远端服务器的文件/目录传输，超级赞
  - Git
  - file header 自动给各种代码文件加上头部的一些固定声明以及正文代码模板(可以自定义哦)，非常有用！
  - Anaconda 可以自动调整格式，让 python 代码符合 pep8 规范
  - sublimelinte pep8 让你的 python 代码更加规范
  - allautocomplete 在全部打开的标签页面中搜索关键字来补全
  - phpintel
  - sublimeREPL 各种语言的 REPL 直接在 sublime 中像文件一样打开，包括 shell, 简直神器; 结合分屏使用，就能一边写代码，一边运行了。
  - bracket highlighter 括号高亮，解决括号失配问题
  - traling space 消灭多余空格
  - docblock 注释如此简单, /\*\* 回车即可
  - origami 多分屏; 和 tmux 的用法有点像，用前缀键(ctrl k), 真心酷毙了
  - [javatar](https://javatar.readthedocs.org/en/latest/) 将 sublime 变为 java 开发环境, ctrl shift k 按两次打开 javatar 命令板.
  - [ensime](https://github.com/ensime/ensime-sublime)
  - [HTML-CSS-JS Prettify](https://packagecontrol.io/packages/HTML-CSS-JS%20Prettify): 代码格式美化
  - advaced csv 对 csv 格式文件有更好的支持
  - markdown editing
  - cmake snippets CMAKE 命令的补全提醒
  - [CppBuilder](https://packagecontrol.io/packages/CppBuilder) 主要用在 C++项目的管理，好像并不很必要

* sublime shotcuts

  ctrl <方向键>， ctrl shift <方向键>
  ctrl pagedown/pageup : 切换 tab
  Ctrl G: go to
  Ctrl R: reach
  ctrl d: 选择单词, 连续多选，ctrl k 放弃当前选择（跳过）

  ctrl kk
  ctrl k b
  ctrl ku, ctrl kl
  ctrl 0: go to sidebar, 侧边栏上用左右方向键选择展开和收起目录

  ctrl [, ctrl ]

  shift 大写

  ctrl enter: 行后插入

  ctrl shift enter 当前行之前一行插入

  ctrl shift d: 复制整行

* sublime 更多使用

  1. 可以自定义一些 snippets, 用 ctrl shift p 就能检索，充分利用 sublime 的扩展能力
  2. project
  3. 开启 Vim 模式

- vim

  用如下命令就能将 vim 配置成小钢炮

  > wget -qO- https://raw.github.com/ma6174/vim/master/setup.sh | sh -x

* tmux

  绝对神器。

* [Cmd Markdown](https://www.zybuluo.com/mdeditor#)

### Chrome 插件

- vimium

  将 chrome 打造成 Vim，诸多快捷键，浏览网页不再用鼠标
  注：？显示快捷键

- RSS feed Reader
- Pocket 　资源收藏就用它
- google mail checker
- WizNote Web Clipper
- Web Timer
- 划词翻译
- SwitchOmega
- AdBlock
- Wunderlist
- kami

### CDN

- [百度静态资源公共库](http://cdn.code.baidu.com/)

### Presentation

- [Impress.js]() 可以替代 Prezi, 毕竟 Prezi 太贵了
- [Jmpress.js](http://jmpressjs.github.io/jmpress.js/#/home) jQuery 版的 impress.js
- [deck.js]()
- [Reveal.js](http://lab.hakim.se/reveal-js/#/)
- [Remark.js](http://remarkjs.com/)

### 翻墙

没有 google, 我真心活不下去----

goagent 太折腾，曲径/红杏会挂，还是自己买个 VPS 用 shadow socks 搭建一个比较靠谱。

也可以试试 SS-LINK

### 项目管理

- [Travis CI](https://travis-ci.org/): 开源项目的持续集成环境。

### 其他

- git

  [Pro Git](https://git-scm.com/book/en/v2) is all you need to learn git.

  [git 钩子](https://github.com/geeeeeeeeek/git-recipes/blob/master/sources/Git%E9%92%A9%E5%AD%90.md?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)

- gitbook 　写书　 so easy.
- hexo 搭建静态站点，类似的还有 Jekyll, Octopress 等
- HTML5 页面制作工具　[列表](http://next.36kr.com/posts/collections/61), 包括　[MAKA](http://www.maka.im/home/index.html?utm_source=next.36kr.com)

- OCR

  [tesseract](https://github.com/tesseract-ocr/tesseract): 最早由惠普研发，后开源

  [newOCR](https://www.newocr.com/): 免费在线 OCR 工具

- [HashCat](https://github.com/hashcat/hashcat) 密码破解工具（已开源）

- [微信抢红包插件](https://github.com/geeeeeeeeek/WeChatLuckyMoney?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
