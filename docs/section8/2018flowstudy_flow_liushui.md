[TOC]

## 20180901

- Caucher Birkar
  研究方向是 birational geometry（双有理几何），属于代数几何的范畴，研究的是代数簇在双有理等价之下不变的性质，也就是由其函数域决定的性质。这些性质包括维度、算术亏格、几何亏格、小平维度等等。
  许晨阳研究的也正是这一领域。

  代数簇是代数几何的基本研究对象。

炫酷 h5

http://www.sbs.com.au/theboat/

[Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)

典型的非凸函数，最小值在一个狭长的平谷中，因此也被用来测试优化方法。

[Best Practices for Scientific Computing](https://arxiv.org/pdf/1210.0530v3.pdf)

[中文翻译](http://blog.sina.com.cn/s/blog_a6c971960102uw48.html)在此

其实是科学计算软件编写的最佳实践

科学家们需要自己编写计算软件，但是普遍缺乏软件开发流程，例如代码版本控制、单元测试、任务自动化

> 我们认为，软件不过是另一种实验设备，应当像任何物理设备一样，小心地制造、检查和使用。尽管大多数科学家仔细地检验他们的实验室和现场设备，但是他们却并不知道他们的软件的可靠性。这将会导致严重错误，影响到发表的研究的中心论点：最近出现的引人注目的因为计算方法错误而撤稿或增加技术评论、订正的就包括 Science，PNAS，Journal of Moleclar Biology 等

[I don't like Notebooks ](http://vdisk.weibo.com/s/AoN5oNkBMQwm)

最近正好在大量使用 notebooks, 所以看到这个题目自然很吸引我。

基本上就只提到了一个点： 因为执行顺序和代码顺序可能并不一样，带来理解上的混乱。
但这并不是一个大问题，因为页面上有执行顺序编号，且一般会在整个流程跑完之后，调整为一致顺序。

00 后黑话

zasg 真情实感

ky 没眼力

nbcs nobody cares

nss 暖说说，让人帮忙点评论转发点赞

cdx 处对象

巨巨

290（250+38+2）

政治正确之下，redis 中的 master-slave 改为 master-replica

https://github.com/antirez/redis/issues/5335

Triplet Loss

人脸嵌入
学习一种人脸嵌入，使得同一个人的人脸在嵌入空间中尽量接近，不同人的人脸在嵌入空间中尽量远离

三元组
anchor
positive
negative

FaceNet: A Unified Embedding for Face Recognition and Clustering

层报： 逐层向上报告

长期有耐心

1919 年，毛姆到访中国。
八岁丧母，十岁丧父

> 苍生苦难，不知伊于胡底

## 20180927

应该对新的东西保持尝试的勇气，但也要敏锐的判断其价值，果断放弃无用的东西。

mock server
主要还是为了分工开发的时候测试方便，让接口返回固定的结果，方便测试。
之所以是一个 server， 是因为，当服务比较多的时候，服务之间的关系比较复杂，需要管理。

网易云音乐外链
https://music.163.com/outchain/player?type=2&id=28406914&auto=0&height=66

placebo 安慰剂

## 20180928

CORS 跨域访问
chrome 中打开 html，无论如何无法加载本地 json 文件，即使用 simplehttpserver host 也不行。但是在 safari 中却可以。

## 20180929

github 代码链接，定位到行的方式
https://github.com/tensorflow/tensorflow/blob/799e31f3840c21322e380e1ec6e5bacb95d016fa/tensorflow/examples/saved_model/saved_model_half_plus_two.py#L168

就是在 url 后面加 `#L167` 就能定位到 167 行

## 20180930

> 没有什么东西是真正 transferable 的。如果你在一个项目里真正能够获得什么的话，可能只不过是你对你自己能把一件事情做到多好的自信心而已

将 reeder 正式用起来，回归阅读。

> This is not the end, it is not even the beginning of the end.
> But it is perhaps the end of the beginning.

当前并非结束，并且，远未到结束的时候，但是启动阶段可以说是结束了。

## 20181002

昨夜做梦，梦见机场广播里喊：某某给自己命名为三街坊大葫芦。从梦里笑醒，女友问我笑什么，大笑了好一会儿才能讲给她，她也跟着我大笑。

跟女友聊天，她说自己小时候谁见了都夸长得可爱，但是稍微大点了之后，别人见了就只说：这闺女长的高的，比你妈妈还高哩。大笑。

晚上在比格吃披萨，中途有个老女人，大约是因为服务员收早了她的餐，骂骂咧咧不止，张扬跋扈地啸叫，整个店里都听见了。店长好言相劝，耐心解释赔罪，她还要拉着店长要服务员-一个老大妈-道歉，但是即使老大妈过来道歉了也不依不饶，喋喋不休好一会儿。明明就是自助餐，再取点吃就完了，不知道这个胖丑老女人为啥这么狂拽。。我们坐在边上的人都对这个胖老女人露出鄙夷的神色，除此也无可奈何。

> 我们是离巢之鸟 身下万丈深渊，头上无尽苍穹

[淡豹：我没有生活在一个更美好的时代](https://weibo.com/ttarticle/p/show?id=2309404190265299582395)

> 当某一天我们死去，是我们曾经的劳作定义我们的生命。不是爱，不是牺牲，不是在爱情或家庭中，不是曾经有过的完美关系，或某一段关系中的隐忍和付出——而是那些称之为实践的东西把我们生命的印迹刻在地表，让我们多多少少幸福地死。---劳动创造的价值可能会给人生带来更丰富或更稳定的幸福.

淡豹（真名刘雪婷，和我一个高中同学同名），在'我想拼命踩一踩，那些对女性的偏见' 一文中写道，

> 无处不在的广告宣扬着“女性就爱买买买”，实际上是把女性视为消费者、否定其生产者身份

> 人在能力所及之处不要轻易放弃公共生活，保持一点激进

> 在一个黑暗的时代，不作为就是一种共谋

> 英国女性最高幸福感的年龄是在 85 岁以后，就是当她们熬死了她们的丈夫以后

这种还是作为调侃的段子来理解吧。

保理业务

## 20181007

国庆长假的最后一天了。
回到家中，走的时候收拾东西时弄的一地狼藉纹丝不动，想也没想就关了电闸导致冰箱里的食物发出恶臭。我和女票的生活能力啊---
早上收拾屋子，发现想找个简单的东西，需要翻出一大堆。非常不合理的归置，虽然做到了功能相似的物件大致归到一起，但是基本都是胡乱堆砌到一处，线头缠绕纽结，混乱不堪。收拾了一番，才大致清爽。
还有书的摆放等，拿一本书需要抬开 N 本，真的太不方便，不过这个放到以后归置吧。
我因此反思，自己现在采取的策略应该是有问题的，需要修正。我本以为自己可以放开生活的一切琐碎，专心于工作和事业，这其实是不对的，生活毕竟是两个人的事情，也需要自己上心，两个人一起打磨一个彼此都舒心的环境。

这个长假，头两天分别逛了 798 和动物园，后几天就在西安，参加了一场同学的婚礼，逛了西安交大、钟楼、碑林、城墙、大唐芙蓉园等，吃了肉夹馍、羊肉泡馍、biangbiang 面等，也算不虚此行。唯一遗憾之处是，未能去参观陕博，这也怪自己这次没能提前规划。这也是此行的一个教训，以后凡事总得有个计划，提前了解安排。

木遥的窗子，这个博客真心优质，简直是今年遇到的最佳博客。
学到一个词：端赖（只有依赖）。

> 众所周知，左翼和右翼的概念源自 1789 年法国大革命后国民工会里议员座位的排列方式。这种偶然形成的一维结构具有惊人强大的生命力，几经革命还是一直坚持到了二十一世纪之初。法国虽然小党林立，历次大选最终总还是忠实地归队于左右对决。但这个结构看起来已经无法用来准确刻画今天的世界了。要描述一个典型选民的政治立场，需要用比左右更多的方向才行。
> 所以这个世界是几维的呢？

管制利率

重整河山待后生

> 千里刀光影
> 仇恨染九城
> 月圆之夜人不归
> 花香之地无和平
> 一腔无声血
> 万缕慈母情
> 为雪国耻身先去
> 重整河山待后生

## 20181008

ddl deadline

建设比见证更重要

## 20181010

Orz 是失意体前屈,表示失意或沮丧的心情。还有佩服以及膜拜的意思

奔现 指在网络上认识的两个人由虚拟走向现实发展

饭否网已经停止注册了，但是我突然发现其实我很早就注册过了，今天重置密码登录上了！

用 rsshub 加上即刻，成功将微信公众号文章通过 reeder 订阅，great! 享受慢阅读

Graham Greene 格雷厄姆-格林

## 20181011

libsvm 格式并不是一种好的格式，因为无法区分那些为 0 的特征和缺失的特征，这在 xgboost 这样可以容许缺失值的算法中，是不太好的。

## 20181012

> 没有什么比时间更具有说服力了，因为时间无需通知我们就可以改变一切
> by 余华

家里的马桶盖（准确的说是和马桶盖连在一起的马桶座垫，我也是经过了多次尝试才发现这俩无法拆卸）坏了，拆卸的过程中，一个疑问浮上心头：马桶为什么叫马桶，和马有什么关系？
查了一圈知道，原来最早的时候叫虎子，后来为避讳唐高宗李渊的祖父李虎的名号，改为马桶。

我永远喜欢纸片人，其实是说喜欢钱。

> 熬夜是对一天结束的恐惧

> 我不祝你一帆风顺，我祝你乘风破浪

丧帅

FOBO： Fear of Better Options

> 有时候孤岛不是没有办法连成大陆，人不会是永恒的孤岛，只是还不知道如何发出讯号
> by 马薇薇

> 不要慌，都是小场面

不易青年：不容易，不放弃

tumblr 汤不热，这个翻译真是市井

## 20181013

> 起笔时尚有戏谑，结尾处已至沧桑

> 年岁有加，并非垂老；理想丢弃，方坠暮年。

实物税-一条鞭（需丈量土地）

生活妙招：清除冰箱异味的方法是，放橘子皮，柚子皮，茶叶渣等

> 我爱你不后悔，也尊重故事的结尾

## 20181016

专注于业务，技术应该为业务服务，在此过程中，注意技术的积累即可。

笔画等宽算法

笔画宽度变换 SWT

当前的 ai 系统都是专用系统，这个人的智能也相去甚远。
meta-learning 是

“小熊太可爱了啊。我决定了，我邀请你跟我去厕所”，吕小姐对放在桌上的玩具小熊说道。然后她就抱起小熊奔跳着去了厕所。可爱至极。

## 20181017

http://alteredqualia.com/xg/examples/deepfeature_aging.html 年龄和注视控制，碉堡

重阳节： 奇数为阳，9 是最大的奇数，为极阳，因此为重阳

## 20181018

公牛插座之所以叫“公牛”，是因为创始人喜欢打篮球，而那时候最火的球队是芝加哥公牛

微软联合创始人保罗·艾伦 的标签： SAT 满分，商业奇才， 体育富豪， 探险者，慈善家

美墨边境上两个最大的都市圈：圣地亚哥-蒂华纳，艾尔帕索-华雷斯
华雷斯城和蒂华纳是墨西哥犯罪率最高的两个城市！

Byvoid ： 郭家宝，这个作者写的游记我是真喜欢，和一般的到此一游和流水账不同，充满了文化和历史，真是高级。
byvoid 本人也牛逼。

99% invisible is an independently produced radio show created by Roman Mars that focuses on design and architecture

## 20181019

metatron 梅塔特隆, Metatron 字源于字根 meta-（次于）及 thrones（王座），即代表”最靠近王座者”之意

## 20181021

阿拉-古勒，土耳其的视觉编年史家

ipywidgets 包实现 jupyter notebook 的控件交互

jakevdp (Jake VanderPlas)的博客，非常赞，技术干货。似乎也是 《python 数据科学手册》 一书的作者。

Gizmodo，是美国一个知名科技博客，主要报道一些全球最新的一些科技类产品，比如 iphone 原型机都可能最先出现在这个博客上面
Gizmodo 是最早曝光第四代 iphone 的网站

- 左派和右派的区别
  传统来说，左派通常主张积极改革，主张把旧的意识形态和制度革除，从而建立新的意识形态和制度，反对派一般自认为左派；右派一般较为保守，主张稳妥、秩序、渐进、缓慢的改革方式，强调维护旧有传统，保皇党一般自认为右派。左派与右派都是相对而言，在不同背景下两者主张的具体内容不会相同，不能以静态的“主义”或“阶级”划分。两词是相当广泛的形容用词，用以作为一种广泛的辩证法解释方式，可以用作形容或区分一种政治立场、一种政治意识形态、或是一个政治党派。除非是将其套用至多维的政治光谱上，否则这两词通常是用以描述两种完全相对的立场

1984 年，中国出了一套家庭出身代码。2004 年，因为使用范围较少，被废止。

先胜而后战

## 20181024

有人分析了 medium 上的[百万文章](https://medium.freecodecamp.org/how-i-analyzed-a-million-medium-articles-to-learn-how-many-claps-great-stories-get-d688cf92759f), tag 排名依次是 旅行，诗歌，创业，健康，爱，政治，生产力，教育，领导力，网页开发，js，写作，设计，编程，科技，AI 等等
国外的人文气息太浓厚了吧，尤其是诗歌，居然能排到第二，这个真的服。

还有一个结论是说，只要一篇文章有 2000 个喜欢以上，就是 medium 平台上 1% 的头部文章了。
不过，细分领域的话，在自我提升等 tag 上，需要获得 6661 次喜欢以上，才是该领域的 1%，而诗歌只需要 544

点赞最多的一篇文章: I’m harvesting credit card numbers and passwords from your site. Here’s how.
文章第一句话 The following is a true story. Or maybe it’s just based on a true story. Perhaps it’s not true at all.

怎么做到的，这个家伙居然将恶意代码置于 npm 开源软件包中！

[Travel Is No Cure for the Mind](https://medium.com/personal-growth/travel-is-no-cure-for-the-mind-e449d3109d71)

这个[中东历史](http://bbs.tianya.cn/post-worldlook-1812734-1.shtml)讲得很搞笑，但是也清楚。基本清楚了阿拉伯人和犹太人。

犹太人是旧约，基督教是新约

但是伊朗的波斯人呢

哈里发其实是阿拉伯帝国元首之意

什叶派追随的是穆罕默德的女婿阿里，
逊尼派占了穆斯林的 85%

上帝为什么叫胡大，胡大就是安拉，其实就是基督教和伊斯兰教的不同叫法

## 20181029

要饭没有要早饭的，为什么呢？他能早起就不至于要饭

## 20181031

> 中国的国企改革，差不多就是社会主义市场经济改革的全部。而这句话反过来理解，也可以说，国企改革停滞，差不多就是社会主义市场经济改革的停滞

## 20181105

齿冷：讥笑

## 20181106

苏珊·桑塔格, 一位在西方文化界与西蒙·波伏娃齐名的女性

## 20181107

`Bidirectional Encoder Representations from Transformers（BERT）`

> 预训练表示可以是与上下文无关的或与上下文相关的。与上下文相关的表示又可以分成单向（只跟上文或下文相关）或双向的 （同时考虑上文和下文）。与上下文无关模型诸如 word2vec 或 GloVe 之类的对每个单词生成一个词嵌入表示，所以在这些模型下 “bank” 一词在 “bank account” 和 “bank of the river” 中会具有相同的表示。而与上下文相关模型则基于句子中其他单词来生成每个词的表示。例如，在句子 “I accessed the bank account” 中，一个单向的上下文相关模型表示 “bank” 会基于上文 “I accessed the” 而非下文 “account”。 然而，BERT 表示 “bank” 会同时使用它的 上文 和 下文 — “I accessed the ... account” — 从深层神经网络的最底层开始，成为双向的表示。

可惜，NERT 需要云 TPU

支持中文以及多语言的预训练基础模型 **BERT-Base**
通过 Colab 和 “BERT FineTuning with Cloud TPUs” 笔记本开始使用 BERT

读 `装物理学家很欢乐很沉重——曹则贤研究员在2017年物理所研究生开学典礼上的致辞 ​`, 真是好文章！

> 医学是一门什么都不确定的科学和什么都可能的艺术

> 我们研究最微观的世界，可能需要的是最宏观的关于整个宇宙的知识。

准晶：一种介于晶体和非晶体之间的固体（人们把固体材料分为两大类，一类是晶体，其中原子作规则排列；另一类是非晶体，原子混乱排列）

> 物理学里面不存在革命，如果你看到了革命，那是因为你知道的少 by 马赫

Fabrice Bellard
法布里斯·贝拉， FFmpeg 和 QEMU(QEMU 的技术已经被应用于 KVM、XEN、VirtualBox 等多个虚拟化项目中) 作者

摘录一段知乎上的[评价](https://www.zhihu.com/question/28388113/answer/150897437)

> 计算机底层功力极其深厚，对各种细节了如指掌，虚拟机可不是想写就能写的，这需要熟悉 CPU、内存、BIOS、磁盘的内部原理，鼠标、键盘、终端等外围设备的工作流程，然后在软件层面模拟出来，想想就复杂。从这一点上他可以被称作天才程序员。另外，他的数学功底也是相当扎实，能发现计算圆周率的新算法并且改进算法的人又可以称作计算机科学家。他一个人几乎涵盖了计算机领域的两大发展路线，属于那种全才型的人物

off limits 禁止入内

op-ed 专栏

[港殇：宗师、黑帮、美人与死亡](https://www.huxiu.com/article/270227.html)
真雄文

林黛原名程月如,乳名却叫和尚。

刘以鬯，香港纯文学第一号人物，活了 100 岁（2018 年去世）

> 生锈的感情又逢落雨天，思想在烟圈里捉迷藏 from 《酒徒》

饶宗颐，业精六学，才备九能

> 要说挣钱，没人比得过李嘉诚；要说读书，没人比得过饶宗颐

mdd cup 2018 冠军分享：匪夷所思的数据分析，扑朔迷离的线下评估，弱不禁风的特征工程。

## 20181108

自动驾驶终将面临道德困境

> Should it avoid a crowd if it means hitting another person? Should it protect passengers over pedestrians? Young over old? Even though such situations will be extremely rare, it is clear that they will sometimes occur.

procrastination 拖延

在强化学习中，如果好奇是奖赏，那机器人一定会在电视机前面不停按遥控器换台。

https://www.youtube.com/watch?v=O1by05eX424&feature=youtu.be 人脸的任意修改，更多头发？更年轻？AI 真的要以假乱真了。

等高线向海拔较高处凸出的是山谷，等高线向海拔较低处凸出的是山脊，因此“凸高为谷，凸低为脊”，简称“高谷低脊”

gauging 测量

carrot-and-stick 胡萝卜加大棒

ppt 特别适合用 微软雅黑字体

## 20181111

双 11

## 20181113

高毅 邱国鹭

## 20181114

less artifical, more intelligient

## 20181115

> 进窄门，走远路，找微光

卤水点豆腐，一物降一物
因为黄豆进过水浸、磨浆、除渣、加热等过程，变成了我们常喝的豆浆，是一种蛋白质胶体。豆浆要想变成豆腐，就必须借助卤水，卤水一点，豆浆中的蛋白质凝聚，就变成了豆腐脑，将里面的水挤压，就变成了豆腐

主要死亡疾病构成

1. 恶性肿瘤
2. 心脏病
3. 脑血管病
4. 呼吸系统疾病

肺癌、胃癌、肝癌、结直肠癌和食管癌排在发病率前五位。（这也是男性的排序）
乳腺癌是中国女性肿瘤发病首位！

Druid
阿里巴巴开源平台上一个数据库连接池实现

## 20181116

海屋添筹的典故，出自《东坡志林》里的故事：三个老人在那里论年龄，一个说我不记得自己多大了，只记得小时候认识盘古；一个说我也不记得了，只知道我住在海边，看到沧海变桑田，我就往屋子里扔一个竹筹，现在已经堆满十个屋子了；最后一个说，我更不记得自己几岁了，只知道每次去吃蟠桃会的蟠桃，把核扔昆仑山下，现在堆的和昆仑山一样高了。
海屋添筹，就是第‘二个老人的故事。甲子，就是天干地支纪年一个轮回，60 年。
华封三祝的典故出自《庄子》，尧曾经巡视天下，在华地得到守卫者的祝愿，三个祝愿中的第二个就是祝愿长寿。而老人星，就是寿星，或者叫南极星，也就是大家都知道的南极仙翁的星位

## 20181119

> 我才知道我的全部努力，不过完成了普通的生活

## 20181120

斯巴鲁力狮
保养比较贵

## 20181126

> 买学区房这件事情，再次反映了人类很典型的一种思维模式——简化。
> 我们通过星座、血型等给别人贴标签，以简化自己通过深入接触而了解别人的过程。
> 我们通过买学区房来简化教育孩子所需要的投入，同时来缓解家长对于儿女教育问题的焦虑。

OKR 工作法
Intel 的传奇人物安迪·格鲁夫
OKR 即 Objective 与 Key Results 的首字母缩写。
制定一个较长期的目标（Objective），并且将目标分解成为一些关键的结果（Key Results）

OKR 在这个核心思想下，再附加有一些规范：

希望目标和关键结果不要泛滥，小的团队和个人刚开始最好只定一个目标，3 个以内的关键结果，以便再加专注。

- OKR 的实施结果不要与绩效挂钩。不要与绩效挂钩。不要与绩效挂钩。希望我重复三遍能有一些强调效果。
- OKR 的目标应该较长，最少一个月以上。我个人觉得以季度为粒度的阶段目标较为合适。
- OKR 不追求完成，而是追求最大挑战。关键结果应该制定在 50% 可能性能够完成的假设上，以便发挥团队最大的动力。稍后我们看到，我们用 5/10 来表示初始的关键指标信心指数。
- 实施 OKR 的团队应该是自组织的团队，而不应该严重依赖于外部团队成员。
- OKR 的目标和关键结果应该由团队讨论决定，而不应该是领导指定。
- 每隔一段时间回顾一下 OKR 的结果，大家分享成绩，并重新调整信心指数。

## 20181130

stackoverflow 2018 开发者年度调查结果（超过 10 万名开发者参与）
https://insights.stackoverflow.com/survey/2018#most-loved-dreaded-and-wanted

来自中国的参与者只有 1%，而印度是 13%

差不多 60% 的都是后端开发，但是最受欢迎的框架却是 tensorflow

超过 50% 的人有 2 个显示器

## 20181203

https://medium.com/s/story/7-things-you-need-to-stop-doing-to-be-more-productive-backed-by-science-a988c17383a6

Sometimes, working less can actually produce better results.

The key to success is not working hard. It’s working smart.

Stop doing everything yourself and start letting people help you

无尽的硝烟：医改 15 年拉锯战
https://weibo.com/ttarticle/p/show?id=2309404311986342093081

邓云乡

虽苦不悲

Bardo 是佛教名词“中阴身”的藏语写法，指人死后到转世前的那一段状态

为稻粱谋

## 20181206

不待扬鞭自奋蹄

## 20181207

> 我这样的穷人跟你赌，不是看你要什么，而是看我有什么

## 20181211

Calabi-Yau Landscape

在代数几何中，一个 Calabi-Yau 流形，或者称为 Calabi-Yau 空间，是一种具有 Ricci-flatness 特性的流形, 在理论物理中有应用，特别是在超弦理论中：时空的额外纬度被认为是 6 维的 Calabi-Yau 流形。

Ricci-flat 流形

这里的 yau 是 Shing-Tung Yau ，即丘成桐

## 20181213

伟大的背后都是苦难

jupyter notebook 的一大特点是：反馈几乎是实时的，也就意味着测试是实时的

https://towardsdatascience.com/3rd-wave-data-visualization-824c5dc84967

## 20181214

https://towardsdatascience.com/forensic-deep-learning-kaggle-camera-model-identification-challenge-f6a3892561bd

因为每种照相机，它背后对图片的后处理都是有一定程式的，因此可以利用此来判断一张图片是何种照相机拍摄，这在刑侦等领域有重要的应用。

如何判断是新的照相机？
这篇文章写的真不错

文中提到的 [cyclic learning rate](https://medium.com/@lipeng2/cyclical-learning-rates-for-training-neural-networks-4de755927d46)

Stochastic gradient descent with restarts (SGDR)

checkpoint averaging

## 20181222

反脆弱

> 脆弱如此可怕，那么，脆弱的反面是什么？是坚强、坚韧吗？脆弱的反面并不是坚强或坚韧。坚强或坚韧只是保证一个事物在不确定性中不受伤或保持不变，却没有办法更进一步，让自己变得更好。而脆弱的反面应该完成这个步骤，不仅在风险中保全自我，而且变得更好、更有力量。做一个类比的话，坚强或坚韧就像一个被扔到地上的纸团，不会摔坏，但是也只是维持了原貌，这还不够。

> 和纸团相反，乒乓球扔到地上非但不会摔坏，反而可以弹得更高。乒乓球拥有的就是脆弱反面的能力，也就是反脆弱

## 20181224

学会向别人解释一件事，尽管它看起来简直完全无需解释。

并非是一个牛逼哄哄的想法和方案才值得提出，大多数时候，一个简单的（但需考虑的相对全面）方案就已经满足需要，不要鄙视基本和简单的东西，掌握好它们。

CDMA 是码分多址的英文缩写(Code Division Multiple Access)，它是在数字技术的分支——扩频通信技术上发展起来的一种崭新面成熟的无线通信技术

## 20181226

> 有何胜利而言，挺住就意味着一切

## 20181228

花时间看组内周报

在公司强 wiki 环境下，幕布越来越显得不合时宜。2019 年，工作部分的幕布，先不用了吧还是。

python 以后强制使用 python3

> 不教而杀谓之虐，不戒视成谓之暴，慢令致期谓之贼
