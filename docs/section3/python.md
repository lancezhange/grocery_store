# Python

[[TOC]]

## 语言特性

> 所有的变量都可以理解是内存中一个对象的“引用”

## 环境

`brew upgrade python # 安装 python3`

```
Python has been installed as
  /usr/local/bin/python3
3.7.3


Unversioned symlinks `python`, `python-config`, `pip` etc. pointing to
`python3`, `python3-config`, `pip3` etc., respectively, have been installed into
  /usr/local/opt/python/libexec/bin

If you need Homebrew's Python 2.7 run
  brew install python@2

Pip, setuptools, and wheel have been installed. To update them run
  pip3 install --upgrade pip setuptools wheel

You can install Python packages with
  pip3 install <package>
They will install into the site-package directory
  /usr/local/lib/python3.7/site-packages
```

`/Users/zhangxisheng/anaconda/bin/jupyter notebook` 用 python2 作为主环境的 jupyter notebook

`brew info python`
会发现，我的机器上其实有 2.7， 3.6 和 3.7 多个环境！

将 python 3.6 路径加入 path 之后，python3 又可以关联到 python 3.6 了！
但是 pip3 还是 python 3.7 的（通过 pip3 --version 可以看到）

那么，python3.6 如何安装包呢

`python3 -m pip install --upgrade sacredboard`
or
`alias pip3='python3 -m pip'` (但是 sudo pip3 依然用的是系统的 3.7)

因此，以后==全部用 anaconda 版本的==

下载速度慢，换源
`python3 -m pip install --upgrade scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple` 用清华的源

```python
PACKAGE_DIR = '/Users/zhangxisheng/github_opensouced_project/deep-ctr-prediction'
sys.path.insert(0, PACKAGE_DIR)
```

## 日志

log

```python
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
```

try:
except:

## 基本数据结构

1. **带有默认值的字典**

```python
from collections import defaultdict
d = dict()
d = defaultdict(lambda: -1, d)
d['a']
```

2. list comprehension
   `t = [True if (i == 'n' or i == 'v') else False for i in x.split("/")]`

3. flatMap 的一种实现：利用 reudce 函数
   reduce(list.**add**, [i.split(",") for i in x])
   或者较为复杂的 list comprehension
   flattened_list = [y for x in list_of_lists for y in x]

.lower() 小写

叉积，组合，交叉
for p in itertools.product([1, 2, 3], [4, 5]):
print(p)

for p in itertools.combinations([1,2,3],2):print(p)

flattern

```python
import itertools

a = [[1, 2], [3, 4], [5, 6, 7]]

list(itertools.chain.from_iterable(a))

sum(a, [])
```

```python
a = "abcdef"
it = iter(a)

next(it)

print(*it)  # 剩余的元素
```

集合中随机选取元素
import random
for i in random.sample(poi_tag, 2):
print i, poi_tag[i]

#### 魔力方法/属性

我们知道，python 中，**一切皆对象**，同时，python 也是一门多范式的语言：面向过程/面向对象/函数式和谐共存，这背后的奥秘就是 **magic method**.

事实上，许多运算符和内置函数都是用魔力方法实现的。例如，_+_ 其实是 `__add__()`, `len()` 为 `__len__()`，而任何具有`__call__()` 方法的对象都被当做是函数。此外，`with`**上下文管理器**也是借助`__enter__()` 和 `__exit__()`魔力函数实现的。

如下列举一些重要的魔力方法和魔力属性

1. `__dict__` : 属性列表，`object.attr` 其实就是 `object.__dict__[attr]`，许多内置类型，如 list, 都没有该属性；类的属性
2. `__class__`, `__bases__`,`__name__`,
3. `__slots__` 对拥有该属性的类的对象，只能对`__slots__`中列出的属性做设置
4. 用以包构建的，如 `__all__`
5. `__setitem__`
6. `__init__()`,`__new__`

一个类的实例像函数一样被调用，就要借助 **call**

**init** 其实是初始化方法，真正的构造方法是 **new**
可以说，**new** 是个 static class method, 而 **init** 是 instance method

    Use __new__ when you need to control the creation of a new instance.
    Use __init__ when you need to control initialization of a new instance.*

1. `__repr__`, `__str__`
2. `__exit__(self, type, value, traceback)`
3. `__iter__`(通常和 `yield` 一起用以定义可迭代类)
4.

class Foo(object):
def **init**(self, a):
print "run in init"
print "a = ", a

    def __call__(self, *args):
        print "run in call"
        print "args = ", args

f = Foo('ha') # 调用 init
f("hb","hd") # 调用 call
可见，**call** 等价于重载了括号运算符。

中文字符串截取
x.decode('utf8')[0:n].encode('utf8')

中文字符

```python
# -*- coding: utf-8 -*-

import re

s = u"中华人民333dffg"
match = re.match(ur"[\u4e00-\u9fa5]+", s)
if match:
    print match.group(0)

```

二维数组遍历
`','.join(str(item) for innerlist in a for item in innerlist)`

#### 动态类型

对象是存储在内存中的实体，程序中用到的名称不过是对象的引用。**引用和对象分离**
乃是动态类型的核心。引用可以随时指向新的对象，各个引用之间互相独立。

可变数据类型，如列表，可以通过引用改变自身，而不可变元素，如字符串，不能改变引用对象本身，只能改变引用的指向。

函数的参数传递，本质上传递的是引用，因此，如果参数是不可变对象，则对参数的操作不会影响原对象，这类似于 C++中的值传递；但如果传递的是可变参数，则有可能改变原对象。

#### lamda 函数

我最早是在 haskell 中见到匿名函数的，后来它被加入到了 python 以及 java 中。python 中定义 lamda 函数很简单：

        func = lambda x,y: x+y

其他函数式编程的经典函数如 map, filter, reduce 等，我最早也是在 haskell 中见到的。大多类似，不再赘述。

#### 迭代器

- 循环对象 例如 open() 返回的就是一个循环对象，具有 `next()` 方法，最终举出 `StopIteration` 错误
- 迭代器 和循环对象没有差别装饰器
- 生成器 用以构建用户自定义的循环对象，这要用到神秘的`yield`关键字

#### 装饰器

函数装饰器接受一个可调用对象作为参数，并返回一个新的可调用对象。
装饰器也可以带有参数，从而更为灵活。

类装饰器同理。

例如，上下文管理也可以用 _contexlib_ 模块用装饰器的方式实现。

- @statcimethod

- @classmethod

- @property

```python
class decorator(object):
    def __init__(self, f):
        print "run in  decorator init"
        self.f = f

    def __call__(self, a, b):
        print "calling deperator"
        return self.f(a, b) + 12

@decorator
def add(a, b):
    print ("original a+b = ", a + b)
    return a + b
```

上面这个例子中，是 class 作为装饰器，装饰了函数
反过来，函数也可以用来装饰 class，例如，下面的 singleton 机制：

```python

def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class MyClass:

```

更多装饰器，参见 [awesome python decorator](https://github.com/lord63/awesome-python-decorator)

反射

```python
m = import_module("src.modelzoo.ModelAnimal")

cls = getattr(m,"ModelAnimal")

c = cls(2)

c.func()

```

编码

已经是 unicode 的字符串，需要原生 unicode 转义

```python
x = "\u676d\u5dde"
x
# '\\u676d\\u5dde'
print x
# \u676d\u5dde
x = x.decode('raw_unicode_escape') # 等价于赋值的时候前缀u,例如： x= u"\u676d\u5dde"
x
# u'\u676d\u5dde'
print x
# 杭州
```

对于中文，正则

```python
import re

pattern = re.compile(ur'([道|路]+)([0-9]+)号')

import sys
reload(sys)
sys.setdefaultencoding('utf8')


line = "清华道2号"
line = unicode(line)
match = pattern.match(line)
```

### 宏

> MacroPy provides a mechanism for user-defined functions (macros) to perform transformations on the abstract syntax tree (AST) of a Python program at import time. This is an easy way to enhance the semantics of a Python program in ways which are otherwise impossible, for example providing an extremely concise way of declaring classes.

下面这段代码是怎么实现的？

```python
from macropy.tracing import macros, trace
with trace:
    sum = 0
    for i in range(0, 5):
        sum = sum + 5
```

MacroPy

#### 闭式

闭式可以减少定义函数时的参数，例如可以利用闭包来定义泛函。

### 元类(metaclass)

我们说过，python 中一切皆对象，就连类也是对象（Classes are objects too）！
那么，类是哪个类的对象呢？答案是 元类

metaclass 是用来创建类对象的类。`__class__` 的 `__class__` 为 `type`, type 是 python 中内置的创建类的元类。
type is the metaclass Python uses to create all classes behind the scenes.

像 int，str，function ，其实都是 class
type 最常见的还是看类型，但其实还能创建类，可谓身兼数职！这也是 python 语言中比较少见的的一个令人难以相信的东西。

通过 type 创建类的一般格式

type(name of the class,
tuple of the parent class (for inheritance, can be empty),
dictionary containing attributes names and values)

因此，如下类的定义
class Foo(object):
... bar = True
其实可以写为
Foo = type('Foo', (), {'bar':True})

这样的话，创建动态类简直易如反掌！

自定义元类： `__metaclass__`，可以被赋值为任意可调用对象

说了这么多，还是不明白什么是元类，元类有什么用？别着急，看[这里](https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python)

元类的创建：继承 type

A metaclass is most commonly used as a class-factory.

### 鸭子类型

例如

def echo(s):
print s.str

上面这个例子中，echo 函数的传入参数，只要可以取 str，就都可以，没有限制 s 的具体类型。

#### 描述符

参考 [python 描述器引导](http://pyzh.readthedocs.io/en/latest/Descriptor-HOW-TO-Guide.html)

`__get__`, `__set__` , `__delete__` ： 实现这三个方法中任意一个的对象叫做描述器

默认对属性的访问控制是从对象的字典里面(**dict**)中获取(get), 设置(set)和删除(delete)它
举例来说， a.x 的查找顺序是, a.**dict**['x'] , 然后 type(a).**dict**['x'] , 然后找 type(a) 的父类(不包括元类(metaclass)).如果查找到的值是一个描述器, Python 就会调用描述器的方法来重写默认的控制行为

注意, 只有在新式类中时描述器才会起作用

#### 继承

1. super()
2. MRO(method resolution order): 也就是通常说的继承顺序，并非简单的深度或者宽度优先，而是确保所有父类不会出现在子类之前
3. `self`, `cls`

#### 协程

协程，用户级线程，可以`让原来要使用异步+回调方式写的非人类代码,可以用看似同步的方式写出来`。具体地，它可以保留上一次调用的状态，和线程相比，没有了线程切换的开销。

`yield`, `next()`, `.send()`, `.close()`

#### 并行

#### 自省(introspection)

`dir`, `type`, `id`

`inspect` 模块

#### 性能优化建议/debug

debug： *pdb*模块

[官方 Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)

##### C 接口

1. CDLL() 加载动态库，和 R 中的做法很相似。
2. C API

### 单元测试

[Python 单元测试和　 Mock 测试](http://andrewliu.in/2015/12/12/Python%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E5%92%8CMock%E6%B5%8B%E8%AF%95/)

#### 其他

1.  使用 _LBYL_(look before you leap, 例如前置 if 条件判断) 还是 _EAFP_(easy to ask forgiveness than permission，例如 `try--catch`)? 这里给出了一个[建议](http://stackoverflow.com/questions/5589532/try-catch-or-validation-for-speed/)

2.  不定长无名参数 `\*args`(元组) 和 不定长有名参数 `\*\*kwargs`（列表）

3.  `python -m SimpleHTTPServer 8088` 文件共享从此 so easy.
4.  `self` 并非关键字，而只是一个约定俗成的变量名

dict 的方法

shop_id = shops.main_poi_id.values
n_pois= shops.n_pois.values
tag_map = dict((name, value) for name, value in zip(shop_id, n_pois))

注意，字典是 dict， 不是 map

assert len(data.poi_id.unique()) == data.shape[0], "门店有重复！"

python3 环境
conda create -n python3 python=3 anaconda
(我的路径为 /Users/zhangxisheng/anaconda/envs/python3)

(报错： `(eval):5: parse error near`|'`，后来查明，是因为我在 sitecustomized.py 中的配置中的干扰，去掉配置之后就可以了)

启动环境: `source activate python3`
这样激活之后，python3 才会打开 anaconda python3

jupyter
ipython notebook 即可启动

jupyterlab 搜索启动

# 快速传入文件

python -m SimpleHTTPServer 8000 注意 windows 上用 powershell

python3 中如下

```python
python3 -m http.server 8000
```

json
`print json.dumps({'4': 5, '6': 7}, sort_keys=True, indent=4)`

## pystyle

```python
pycodestyle --max-line-length=100 --exclude mlflow/protos,mlflow/server/js,mlflow/store/db_migrations,mlflow/temporary_db_migrations_for_pre_1_users -- mlflow tests
pylint --msg-template="{path} ({line},{column}): [{msg_id} {symbol}] {msg}" --rcfile="$FWDIR/pylintrc" -- mlflow tests
```

`pycodestyle` (formerly called pep8) , Python style guide checker

`pycodestyle --show-source`

# 细节

函数必须用 reutrn 返回返回值，如果想有返回值的话

unique 是 函数， unique() 才得到值

忽视警告
import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

忽略命令行下警告错误的输出
python -W ignore yourscript.py

df.values dataFrame 转为 ndarray

y = pd.DataFrame({'poi': x['b'].values, 'deal': x['a'].values, 'bu': x['c'].values, 'prob': x['d'].values}, columns=['poi', 'deal', 'bu', 'prob'])

y.to_csv('/Users/zhangxisheng/Downloads/refund_reason_concerns_refuse_2017-10-19_new.txt', index=False)

Series.values 返回 ndarray

mask
使用()括起筛选条件，多个筛选条件之间使用逻辑运算符&,|,~与或非进行连接，特别注意，和我们平常使用 Python 不同，这里用 and,or,not 是行不通的

data.ix[[231, 236]] 选取行
对索引是整数的，按索引走；对索引是非整数的，按值走；因此，这个函数是应当摒弃的！

既查又改的正确方式
f.loc[f['a'] <= 3, 'b'] = f.loc[f['a'] <= 3, 'b'] / 10

按照 index 列 选取的话，直接 loc 取就行
df.loc[ind].head(100)
如果反向选择，即取反，还得
bad_df = df.index.isin([3,5])
df[~bad_df]

# Select the 5 largest/max delays

delays.nlargest(5).sort_values()
选取 N 最大并不需要全部排序！
delays.nsmallest(5).sort_values()

import heapq
a = [1, 3, 2, 4, 5]
heapq.nlargest(3, range(len(a)), a.**getitem**)

实验记录
sacred
print_config
with arg="value"

降维可视化
hypertools
hyp.plot(sample, n_clusters=2, explore=False, group='label')

时间和日期
datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

计算天数差值
d1 = datetime.datetime.strptime('2017-08-03', '%Y-%m-%d')
d2 = datetime.datetime.strptime('2017-08-02', '%Y-%m-%d')
delta = d1 - d2
print delta.days

<div class="error">Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, but got an instance of 'RangeIndex'</div>

======================================== dataFrame／pandas
df = pd.DataFrame({'B': [4,5,6,7],'A': ['a','b', 'a', 'b'], 'C':['china', 'china', 'eglish', 'egnlish']})

pd.read_csv(data_location, sep=feat_separator, header=None, na_values="null")
读入数据时指定 na 值

箱线图
data[['label', 'hcg_t']].boxplot(by='label')

Pandas 所支持的数据类型:

1. float
2. int
3. bool
4. datetime64[ns]
5. datetime64[ns, tz]
6. timedelta[ns]
7. category
8. object
   默认的数据类型是 int64,float64.

from pandas.util import testing
df = testing.makeDataFrame()

df = pd.DataFrame()

pd.read_clipboard()

df_ta['delivery_duration'] = df_ta['delivery_duration'].astype['float']

df.sample(5) # 随机抽取 5 条数据查看
df.iloc[22] #利用 iloc 命令做行选取

df.loc[22:25] #利用 loc 命令做行选取
df.loc[[22,33,44]] #利用 loc 命令做指定行选取

df['open_int'] = np.nan
df['open_int'] = 999
df['test'] = df.company_type == '民营企业'
df.rename(index=str, columns={'test':'乱加的一列'}, inplace=True) #更改列名，并固化该操作

df_test.drop([2,5],axis=0) #删除行
df_test.drop(['列 1','列 2'], axis=1) #删除列
df.date = df.date.map(lambda x: x.strftime('%Y-%m-%d')) #将时间数据转换为字符串

df['fs_roe'].mean() #计算平均数
df['fs_roe'].idxmax() #返回最大值 index
df.loc[df['fs_roe'].idxmin()]

df.fs_net_profit.corr(df.volume) #求解相关系数
df.company_type.unique() #查看不重复的数据
df_test.dropna(axis= 1) #删除含有 NaN 值的列
m = data.dropna(axis=0, subset=['toshopid']) # 删除某列为 Nan 的行
df_test.volume.fillna(method= 'ffill',limit= 1) #限制向前或者向后填充行数

tmp = tmp.set_index('order_unix_time') 设置索引

group by
文档 http://pandas.pydata.org/pandas-docs/stable/groupby.html#group-by-split-apply-combine
grouped = df.groupby('A')
grouped = df.groupby(['A', 'B'])
grouped.get_group('bar')
按照多列分组
df.groupby(['A', 'B']).get_group(('bar', 'one'))

组内排序
My_Frame['sort_id'] = My_Frame['salary'].groupby(My_Frame['dep_id']).rank(ascending=False)
注意排序的时候，如果是按照多列分组，多列的写法和上面的不太一样：

如果写成 .groupby(samll_shopes['a', 'b']) 则会报错： ValueError: Grouper for '<class 'pandas.core.frame.DataFrame'>' not 1-dimensional

df['Data4'] = df['Data3'].groupby(df['Date']).transform('sum')

### 滑动 卷 往后倒 求平均 周平均

df['A'] = df.rolling(2).mean() 注意，针对的是 index，即 index 每隔两个。由于默认的 index 是从 0 开始的数字，因此和每隔多少行无异
但是如果 index 是 时间戳这种，就可以按照 '2s' 这种间距
df.rolling(2, min_periods=1).sum()
df_re.groupby('A').rolling(4).B.mean() 直接 rolling 会无视索引的区别，但是线分组的话，rolling 也会被限制在索引中
x.B.xs('a').xs('china')

累积
df_re.groupby('A').expanding().sum()

df['new_column'] = pd.Series 这种赋值方式似乎并不是我们想象的那样，例如，当我给一个 df 赋予随机值列的时候，可能因为 df 的 index 不连续，造成好多 NUll 随机数。

df_re.groupby('group').resample('1D').ffill()
resample 等频抽样，D 为天，S 为秒，

### 等频聚合，例如，按周聚合，

df['count'].resample('D', how='sum')

过滤 sf.groupby(sf).filter(lambda x: x.sum() > 2)

df.groupby('g').boxplot()

二维表
pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True)

排序
data_sorted = data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)

箱线图
data.boxplot(column="ApplicantIncome",by="Loan_Status")

mask 注意==判断时，注意每列的类型，不要混淆 int64 和 string

df.xs(index) 按索引选择
如果列名是中文，则用.号取值会报错，用中括号的方式即可。

mask = (x['中国'] <=2 )

import pandas as pd
train = pd.read_csv('train.csv')
train.shape
train.head()
train.columnA.describe()

describe(include='all') 描述所有列（默认只数值列）

取 dataframe 的一列成为 array
shop_id = shops.main_poi_id.values

日期加上一天
dapan['partition_date'] + pd.DateOffset(1)

日期操作函数
df.first_cooperate_month.dt.year.value_counts()

选择和变换
mask = (df_tr.hour.values == 11) | (df_tr.hour.values == 17) & (df_tr.day.values == 17)
注意，.values 取数值， .str 取字符串，
例如 (df.messgae.str.find('model_e') != -1) 此处 find 用于判断字符串查找，或者也可以用 candi[candi['ids'].str.contains("2015122")] 这种方式

但是，但是，对于中文，好像又不能用 str 和 unicode 的 u
例如，tmp = raw_data.loc[raw_data.name == "北京"] 可以，但是 tmp = raw_data.loc[raw_data.name.str == "北京"] 或者 tmp = raw_data.loc[raw_data.name == u"北京"] 都是不对的！！！

对于中文列名，如何处理？
x[x['中国'] <= 2] 这种选择方法当时也是可以的

df['gen'] = df['gen'].mask(df['gen'] + df['cont'] < 0.01)

df.loc[df['First Season'] > 1990, 'First Season'] = 1
df['First Season'] = (df['First Season'] > 1990).astype(int)

选出数值特征
numerric_features = train.select_dtypes(includes=[np.number])
numerric_features.dtypes

筛选非数字特征
categorical = train.select_dtypes(exclude=[np.number])

merge
candi = pd.merge(data, mpoi, left_on=['city', 'geohash_kb'], right_on=['city_name', 'geohash_mt'], how='left')

geohash
import pygeohash as pgh
def geo_encode(x, n=4, by=1):
return str(pgh.encode(x[0]/by, x[1]/by,n)) + "--" + x[2]

按照是否包含在集合中选取 dataframe 符合条件的数据
df = data.loc[data.biz_type.isin({200}) ]
取反 用 ～

分组排序
df.groupby('img')['h'].rank(ascending=False).values

类似的，分组最大
x.r.map(lambda i: x.groupby('r').n.max()[i])

分组计数
x.r.map(lambda i: x.groupby('r').n.count()[i]).values

去重前先排序
df = df.sort_values('lifecycle', ascending=False).drop_duplicates('case_id').sort_index()

变量的协方差矩阵
corr = numerric_features.corr()
协方差最大最小值查看
corr[columnA].sort_values(ascending=false)[:5]
coor[columnA].sort_values(ascending=false)[-5:]

去除重复值 去重
train.columnA.unique()
drop_duplicates(subset=None, keep='first', inplace=False)

数据透视表
pivot = train.pivot_table(index = 某个分类变量, values=某个数值变量, aggfunc=np.mean)
pivot = df_tr.pivot_table(index = 'area_id', values='poi_id', aggfunc=lambda x: len(x.dropna().unique()))
print pivot
还可以加上 columns

多列透视，这个厉害了
p = df_tr.groupby('area_id').aggregate({'delivery_duration':np.mean, 'poi_id':lambda x: len(x.dropna().unique())})

agg 方法将一个函数使用在一个数列上，然后返回一个标量的值。也就是说 agg 每次传入的是一列数据，对其聚合后返回标量。

apply 是一个更一般化的方法：将一个数据分拆-应用-汇总。而 apply 会将当前分组后的数据一起传入，可以返回多维数据。

data.groupby(['model', 'bu']).size()

concat
pandas.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)[source]¶

一列构造多列
df.textcol.apply(lambda s: pd.Series({'feature1':s+1, 'feature2':s-1}))

时间序列
日期类型转换
loandata['issue_d']=pd.to_datetime(loandata['issue_d'])

对此透视表，画条形图
pviot.plot(kind='bar', color='blue')

条件筛选
train = train[train[columnA] < 100]

空值
nulls = pd.dataFrame(train.isnull().sum().sort_values(ascending=false)[:25])
统计每列的 nan 值 或者 缺失值
df.isnull().sum()
缺失值填充
df['time3_category'] = df['time3_category'].fillna(1)
df.fillna(df.mean()['a', 'b']) 这种在数据量很大的时候似乎总是很慢不可用
df[['a', 'b']] = df[['a','b']].fillna(value=0)

fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(df))
imputed_DF.columns = df.columns
imputed_DF.index = df.index

理解 axis
轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第 0 轴沿着行的垂直往下，第 1 轴沿着列的方向水平延伸。

df.mean(axis=1) 计算的是每一行的均值
df.drop("col4", axis=1) 删除的却是这一列的每一行

null 缺失值
xtrain = df.loc[df['Survive'].notnull(), ['Age','Fare', 'Group_Size','deck', 'Pclass', 'Title' ]]
xtrain

SettingWithCop

> A value is trying to be set on a copy of a slice from a DataFrame

```python
from pandas import *
df = DataFrame({"A": [1, 2, 3, 4, 5], "B": [3.125, 4.12, 3.1, 6.2, 7.]})
row = df.loc[0]
row["A"] = 0
```

上面这段代码就会爆错： A value is trying to be set on a copy of a slice from a DataFrame

aggregate 聚合之后，如何重命名聚合列的名称呢？
df = data.groupby('Seed').agg(
{'age':['sum'],
'height':['mean', 'std']})
df.columns = ["_".join(x) for x in df.columns.ravel()]
但是如果聚合函数是 lambda 函数呢？

df['new_column'] = np.multiply(df['A'], df['B'])

设置列名称
null.columns = ['name1']
设置行索引名称
null.index.name = 'name2'

分类变量的值和频次
y = train.columnA.value_counts()
len(y[y>1].unique()) 过滤频次
y = tmp.kb_id.value_counts()
y 这样就能打印出现次数 top 的了，默认排序，很好

value_counts()得到的是 series (with index)， series 转 dataframe:
data.day.value_counts().to_frame()

或者，分组后统计频次，并排序
f = df[['STNAME','CTYNAME']].groupby(['STNAME'])['CTYNAME'] \
 .count() \
 .reset_index(name='count') \
 .sort_values(['count'], ascending=False) \
 .head(5)

哑变量 ont-hot 编码 注意，训练集和测试集要编码一致
train['a_encode'] = pd.get_dummies(train.columnA, drop_first=true)
test['a_encode'] = pd.get_dummies(train.columnA, drop_first=true)

Apply 方法
train.columnA.apply(func)

插补缺失值
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

去除列
data.drop(['featureA'], axis=1)

创建 dataFrame
df = pd.dataFrame()
df['id'] = train.id

dataFrame 转化为 csv
df.to_csv('a.csv', index=False)
d.to_csv('/Users/zhangxisheng/Downloads/big_meal_single_poi.csv', index=False, encoding='utf-8') 有时候有编码问题的时候，加上 encoding 能够解决
存下来的文件，excel 读入之后为乱码的解决方案：encoding='utf_8_sig'

一个 to_csv 的疑难杂症：
我的 df 只有一列，类型为字符串，当我用 to_csv 存储下来之后，发现有些行被加上了双引号，有些则没有。
最后在轩哥的帮助下定位到原因： 这些加上了双引号的行，都是因为本身带有都好；解决办法就是，指明 sep="\t"，这样就行了。

obj.combine_first(other)
如果 obj 中有为 null 的值，用对应索引的 other 中的数据填充

某一列为 nan 的行
df = df[np.isfinite(df['EPS'])]

索引
agg.index.values

列与索引之间可以相互转化
df.set_index('date', inplace=True)
x['index'] = x.index.get_level_values('C')
df.reset_index(inplace=True, drop=False) 对多重索引，将会把每个索引变为列，新加索引为从 0 开始的自然数，非常赞

层次化索引
所谓层次化索引，其实也是多重索引，只不过有些相同的被省略了。
data.unstack() 可以打平层次化索引中的部分索引，即变宽表

多重索引
m_idx = pd.MultiIndex.from_tuples(zip(dates, labels), names=['date', 'label'])
data_dict = {'observation1':obs1, 'observation2':obs2}
df = pd.DataFrame(data_dict, index=m_idx)
参见 http://www.jianshu.com/p/3ab1554fe6f3

类型转换
df_ta['delivery_duration'] = df_ta['delivery_duration'].astype('float')

斜度
train.columnA.skew()

交叉表 cross table
pd.crosstab(df.E,df.group)

标签编码
lbe = LabelEncoder()
lbe.fit(df_tr['area_id'].values.reshape(-1, 1))
df_ta['area_id_le'] = lbe.transform(df_ta['area_id'].values.reshape(-1, 1))

宽表变窄表
df = pd.melt(df, id_vars=["date"], var_name="condition")

窄表边宽表
partition_date bu n_poi
0 2018-04-01 交易销售部 67719
1 2018-04-01 渠道发展部 16774
2 2018-04-01 电销及平台支持部 15683
3 2018-04-02 交易销售部 67526
4 2018-04-02 渠道发展部 16757
5 2018-04-02 电销及平台支持部 15393
6 2018-04-03 交易销售部 67154
7 2018-04-03 渠道发展部 16871
8 2018-04-03 电销及平台支持部 15333
11 2018-04-04 电销及平台支持部 15938

如何转成下面这种表？
日期 交易 渠道 电销
20180401 67719 16774 15683

因为只有下面这种宽表，才能画堆叠的直方图。
一种办法是在 sql 里，聚合一下分别统计成三列就好，这是简单的。
第二种就是： 聚合。
data[['prob', 'model_tag', 'user_id']].pivot_table(values='prob', columns='model_tag', index=['user_id'])

#### loc assign 的用

data.loc[data.penalty_end_day.isnull()]['punished'] = 0 这种不行的
正确的写法： data.loc[data.penalty_end_day.isnull(), ['punished']] = 0

cut 切分和分组
bins = [0, 5, 10, 15, 20]
group_names = ['A', 'B', 'C', 'D']
loandata['categories'] = pd.cut(loandata['open_acc'], bins, labels=group_names)

df['bin'] = pd.cut(df.discount, bins = [0,0.5, 0.7,0.9,1], labels=['low', 'middle', 'normal','high'])
df.groupby('bin').aggregate({'label': np.sum}).plot(kind='bar')

bins = pd.cut(df['Value'], [0, 100, 250, 1500])
df.groupby(bins)['Value'].agg(['count', 'sum'])

cut 和 qcut 的区别
pd.qcut(factors, 5).value_counts()
pd.cut(factors, 5).value_counts()

qcuts 一般会保证每个 bin 中的个数是一样的，而 cut 不会考虑频次，只是平均分值。

分列
grade_split = pd.DataFrame((x.split('-') for x in loandata.grade),
index=loandata.index,columns=['grade','sub_grade'])

df.info(memory_usage='deep') 该表的精确内存使用量，行列个数，以及对应的数据类型个数
在底层，pandas 会按照 数据类型 将 列 分组形成数据块（blocks）
对于包含数值型数据（比如整型和浮点型）的数据块，pandas 会合并这些列，并把它们存储为一个 Numpy 数组（ndarray）。
Numpy 数组是在 C 数组的基础上创建的，其值在内存中是连续存储的。基于这种存储机制，对其切片的访问是相当快的。

看每种类型的块所占内存
for dtype in ['float', 'int', 'object']:
selected = df.select_dtypes(include=[dtype])
mean_useage = selected.memory_usage(deep=true).mean()
mean_useage = mean_useage/1024\*\*2

pandas 中的许多数据类型具有多个子类型，它们可以使用较少的字节去表示不同数据，比如，
float 型就有 float16、float32 和 float64 这些子类型

参考 用 pandas 处理大数据——节省 90%内存消耗的小贴士
https://mp.weixin.qq.com/s?__biz=MzAxNTc0Mjg0Mg==&mid=2653286198&idx=1&sn=f8f0ea4845586b1f9b645995aa07d8a0&open_source=weibo_search

simhash
import re
from simhash import Simhash
def get_features(s):
width = 3
s = s.lower()
s = re.sub(r'[^\w]+', '', s)
return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

print '%x' % Simhash(get_features('How are you? I am fine. Thanks.')).value
print '%x' % Simhash(get_features('How are u? I am fine. Thanks.')).value
print '%x' % Simhash(get_features('How r you?I am fine. Thanks.')).value

print Simhash('aa').distance(Simhash('bb'))
print Simhash('aa').distance(Simhash('aa'))

5.主成分分析

主成分分析是由因子分析进化而来的一种降维的方法。

通过正交变换将原始特征转换为线性独立的特征，转换后得到的特征被称为主成分。主成分分析可以将原始维度降维到 n 个维度。

有一个特例情况，就是通过主成分分析将维度降低为 2 维，可以将多维数据转换为平面中的点，来达到多维数据可视化的目的。

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)

X = pca.fit_transform(data.ix[:,:-1].values)

pos=pd.DataFrame()

pos['X'] =X[:, 0]

pos['Y'] =X[:, 1]

pos['species'] = data['species']

ax = pos.ix[pos['species']=='virginica'].

plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')

ax = pos.ix[pos['species']=='setosa'].

plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)

pos.ix[pos['species']=='versicolor'].

plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax

需要注意，通过 PCA 降维实际上是损失了一些信息，我们也可以看一下保留的两个主成分可以解释原始数据的多少。

6.独立成分分析

独立成分分析将多源信号拆分成较大可能独立性的子成分，它最初不是用来降维，而是用于拆分重叠的信号。

from sklearn import decomposition

pca = decomposition.FastICA(n_components=2)

X = pca.fit_transform(data.ix[:,:-1].values)

pos=pd.DataFrame()

pos['X'] =X[:, 0]

pos['Y'] =X[:, 1]

pos['species'] = data['species']

ax = pos.ix[pos['species']=='virginica'].

plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')

ax = pos.ix[pos['species']=='setosa'].

plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)

pos.ix[pos['species']=='versicolor'].

plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)

Out[42]:

<matplotlib.axes.\_subplots.AxesSubplot at 0x7f47f274af28>

7.多维尺度分析

多维尺度分析试图寻找原始高维空间数据的距离的良好低维表征。

简单来说，多维尺度分析被用于数据的相似性，它试图用几何空间中的距离来建模数据的相似性，即用二维空间中的距离来表示高维空间的关系。

数据可以是物体之间的相似度、分子之间的交互频率或国家间交易指数，而且是基于欧式距离的距离矩阵。

多维尺度分析算法是一个不断迭代的过程，因此，需要使用 max_iter 来指定较大迭代次数，同时计算的耗时也是上面算法中较大的一个。

from sklearn import manifold

from sklearn.metrics import euclidean_distances

similarities = euclidean_distances(data.ix[:,:-1].values)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed",

n_jobs=1)

X = mds.fit(similarities).embedding\_

pos=pd.DataFrame(X, columns=['X', 'Y'])

pos['species'] = data['species']

ax = pos.ix[pos['species']=='virginica'].

plot(kind='scatter', x='X', y='Y', color='blue', label='virginica')

ax = pos.ix[pos['species']=='setosa'].

plot(kind='scatter', x='X', y='Y', color='green', label='setosa', ax=ax)

pos.ix[pos['species']=='versicolor'].

plot(kind='scatter', x='X', y='Y', color='red', label='versicolor', ax=ax)

=============================== 正则匹配统计
import re

pattern = re.compile(r'(._)\_ck(._)\_(.\*)')
cks = {}

with open("hyclicktags.csv", "r") as f:
for line in f:
tmp = int(line.split(",")[1])
match = pattern.match(line)
if match:
if match.group(2) in cks:
c = cks.get(match.group(2))
cks[match.group(2)] = c + tmp
else:
cks[match.group(2)] = tmp

count = len(cks)
print count

if count <= 200:
for i in cks:
print i, cks[i]

带有中文的正则
import re

pattern = re.compile(ur'([道|路]+)([0-9]+)号')
line = u"清华道 2 号"
match = pattern.match(line)

if match:
print match.group(0)
print match.group(3)

print pattern.search(line).group()

正则替换
例如，替换括号中的内容
import re
str = '多摩君 1（英文版）c(ab)(34) '
out = re.sub('（._?）|\(._?\)', '', str)
print out

import re
str = '多摩君一区 '.decode('utf8')
out = re.sub(u"[一区]|[二区]", '', str)
print out

### 推荐这种方式

import re
def remove_district(x):
return re.sub(u"[一区]|[二区]", '', x).encode('utf8')
print remove_district(u'武汉一区')
该函数可以直接用在 df 的列变换中

执行字符串语句
x = eval("2+2")

加载模块的另一种方式

file, path_name, description = imp.find_module('env', [dir])

# 这一步就是导入 env 这个模块，让 B 成为 A 类的别名

B = imp.load_module('env', file,path_name, description).A

最频繁元素
from collections import Counter

def Most_Common(lst):
data = Counter(lst)
return data.most_common(1)[0][0]

在 python 中，True == 1, False == 0， 因此，在列表中是可以用 True/False 做下标取值的，例如：
[exponential(12), exponential(2)][uniform() < 0.5] 相当于每次随机取一种分布的随机数

## numpy

numpy 101 题 https://www.machinelearningplus.com/101-numpy-exercises-python/

去重的时候顺便统计频次
(values,counts) = np.unique(p,return_counts=True)
ind=np.argmax(counts)
print values[ind], counts[ind]

import numpy as np
np.log() 对序列做对数变换
np.exp()

每行的和
df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
注意： 对 series ， apply 的方法，不能写 axis=1

每列的和
df.loc['Row_sum'] = df.apply(lambda x: x.sum())

快速拼接两列
df["dh"] = df["partition_date"].map(str) + '-'+ df["hour"].map(str)

替换
arr[arr % 2 == 1] = -1

where 筛选
x = np.where(arr % 2 == 1, -1, arr)

重复
b = np.repeat(1, 10).reshape(2,-1)

np.r\_[np.repeat(a, 3), np.tile(a, 3)]

公共元素
np.intersect1d(a,b)

# From 'a' remove all of 'b'

np.setdiff1d(a,b)

np.where(a == b)

index = np.where((a >= 5) & (a <= 10))
a[index]

标量函数向量化
pair_max = np.vectorize(maxx, otypes=[float])

boolean to int
y_pred = (y_pred >= 0.5)\*1

# scikit-learn

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size = 0.33)

线性模型
from sklearn import linear_model
lr = linear_model.LinearRegression()

训练模型
model = lr.fit(x_train,y_train)

model.score() 返回被模型解释的方差的占比（即 R 方）

模型预测
predicts = model.predict(x_test)

模型评估/评测
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, predicts)

岭回归
ridge = linear_model.Ridge(alpha=0.5)

ridge_model = ridge.fit(x_train, y_train)

one hot 编码
单列 OneHotEncoder(sparse = False).fit_transform( testdata[['age']] )
多列 OneHotEncoder(sparse = False).fit_transform( testdata['age', 'salary'])
OneHotEncoder 不能对字符串型的值做处理, 因此需先用 LabelEncoder 对离散特征编码

LabelBinarizer().fit_transform(testdata['pet'])

==================== csv 文件处理

import pandas as pd

from datetime import datetime

data_location = "/Users/zhangxisheng/Documents/projects/商家不接待项目/refuse_samples.csv"

df = pd.read_csv(data_location, sep=',', header=0)

def str2date(date_str, formats="%Y-%m-%d %H:%M:%S"):
return datetime.strptime(date_str, formats).date()

dt.weekday() 返回一周的第几天，注意，周一是 0

df["C"] = "" 空列

df['date'] = df['date'].map(lambda x: str2date(x))

print df.shape

print df.head(10)
print df['date'][:3]

print df['date'].value_counts()

聚合
print pd.pivot_table(df, values='fea#1', index=['fea#5'], columns='label', aggfunc=len).reset_index()

多列转一列 由多列合成一列
df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)

列求和
small.sales.sum()

shuffle 混洗
from sklearn.utils import shuffle
df = shuffle(df)

划分的时候不要混洗
train_test_split(y, shuffle=False)

### ============= sklern

有时候将 dataFrame 用 df.values 转为 ndarray 进行训练的时候，爆出如下错误
Input contains NaN, infinity or a value too large for dtype('float32')

可以如下检查
if np.any(np.isnan(train_feat)):
print "存在含有 null 值 的列"

        if np.all(np.isfinite(mat)):
            print "存在非有限值的列"

单列作为 ndarray， 要重塑
df.col.values.reshape(-1,1)

排序 pair，指定排序 key
`sorted(model_bst.get_fscore().items(), key=lambda x: x[1], reverse=True)`

救急、自省
查看包路径
from distutils.sysconfig import get_python_lib
print(get_python_lib())

列出 object 及其内存
import sys
ipython*vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('*') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

data.faq.str.startswith('联系', na=False)

按重复次数过滤
data.groupby('case_id').filter(lambda x: len(x) >= 2)

np.meshgrid
https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python-numpy 看了这个解释你还不知道这个函数是干啥的你打我！

从 url 读取图片

```python
def detect_and_draw(img_url):
    url = "http://10.20.42.6:8415/TextDetect"
    r = requests.post(url, data=img_url, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    result = r.json()['text']


#     img = Image.open(StringIO(urllib.urlopen(img_url).read()))
    img_file = cStringIO.StringIO(urllib.urlopen(img_url).read())
    img = Image.open(img_file)

    img_data = np.array(img, dtype=np.uint8)

    fig,ax = plt.subplots(1)
    ax.imshow(img_data)

    print "%d boxs detected" %(len(r.json()['text']))
    for i in r.json()['text']:
        x = int(i['x0'])
        y = int(i['y0'])
        w = int(i['x1']) - int(i['x0'])
        h = int(i['y1']) - int(i['y0'])
        rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()
```

假设检验

from scipy.stats import binom_test
binom_test(2, 8, 11/2364, alternative='greater')

# 动画

```python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
patch = plt.Circle((5, -5), 0.75, fc='y')

def init():
    patch.center = (5, 5)
    ax.add_patch(patch)
    return patch,

def animate(i):
    x, y = patch.center
    x = 5 + 3 * np.sin(np.radians(i))
    y = 5 + 3 * np.cos(np.radians(i))
    patch.center = (x, y)
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=20,
                               blit=True)

plt.show() # 如果要在 jupyter notebook 中显示，则改为 `HTML(anim.to_jshtml())` 即可；不过，在 jupyter notebook 中显示动效真的很慢！
```

动画甚至可以保存为 mp4 动画

```python
anim.save('/Users/zhangxisheng/Downloads/animation.mp4', fps=30,
          extra_args=['-vcodec', 'h264',
                      '-pix_fmt', 'yuv420p'])
```

# 输入框

```python

from Tkinter import *

def printData(firstName, lastName):
    print(firstName)
    print(lastName)
    root.destroy()

def get_input():

    firstName = entry1.get()
    lastName = entry2.get()
    printData(firstName, lastName)


root = Tk()
#Label 1
label1 = Label(root,text = 'First Name')
label1.pack()
label1.config(justify = CENTER)

entry1 = Entry(root, width = 30)
entry1.pack()

label3 = Label(root, text="Last Name")
label3.pack()
label1.config(justify = CENTER)

entry2 = Entry(root, width = 30)
entry2.pack()

button1 = Button(root, text = 'submit')
button1.pack()
button1.config(command = get_input)

root.mainloop()

```

关于 tkinter

pack(side=TOP) 从上往下排列，而 side=LEFT 是从左往右排列

单选和复选框

```python
from tkinter import *

class Radiobar(Frame):
    def __init__(self, parent=None, picks=[], side=TOP, anchor=W):
       Frame.__init__(self, parent)
       var = IntVar()
       self.var = var
       i = -1
       for pick in picks:
        i = i+1
        chk = Radiobutton(self, text=pick, variable=var, value = i)
        chk.pack(side=side, anchor=anchor, expand=YES)

    def state(self):
      return self.var.get()


class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      self.vars = []
      for pick in picks:
         var = IntVar()
         chk = Checkbutton(self, text=pick, variable=var)
         chk.pack(side=side, anchor=anchor, expand=YES)
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)


if __name__ == '__main__':
   root = Tk()
   # lng = Checkbar(root, ['Python', 'Ruby', 'Perl', 'C++'],side=TOP)
   # lng.pack(side=TOP,  fill=X)
   # lng.config(relief=GROOVE, bd=2)
   labels = [u'类别', u'菜名和价格', u'菜名', u'价格', u'标题', u'标签', u'规格', u'其他文字', u'非文字']
   lng = Radiobar(root, picks = labels)
   lng.pack(side=TOP,  fill=X)
   lng.config(relief=GROOVE, bd=2)

   def allstates():
      print(labels[lng.state()])

   Button(root, text='Quit', command=root.quit).pack(side=RIGHT)
   Button(root, text='Peek', command=allstates).pack(side=RIGHT)
   root.mainloop()
   print "down"

```

图像可以只用用 Image.open 读取； 如果要用 cve.read, 注意要转换 rgb 分量

# ======================================================

最地道的 python
`def ecode(x): return 1 if x == 'a' else 0`

python3 中合并字典的方式

```python
x = {'1':4}
y = {'6':6}
{**x, **y}
# {'1': 4, '6': 6}
```

字符串倒序
x = "abcd"
x[::-1]

出现次数最多的元素

```python
t = [12,3,3,4,5,6,7,5,4,4,3,3,3,3,3]
max(set(test), key = test.count)
```

多个条件判断

math,English,computer =90,59,88
if any([math<60,English<60,computer<60]):
print('not pass')

# 通过程序，执行 shell 命令

from subprocess import check_output
print(check_output(["ls", ".."]).decode("utf8"))

dataframe 通过旧列为新列赋值的方法
dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

"v{:04d}".format(x)
四位数字，不足的用 0 补全，格式化

利用装饰器实现统计函数运行时间注解

```python
import time
def timer(fn):
    job_name = fn.__name__
    def new_fn(*args):
        t1 = time.time()
        print('job:' + job_name + ' start')
        result = fn(*args)
        t2 = time.time()
        print('job:' + job_name + ' end, it costs ' + str(t2 - t1) + ' s')
        return result
    return new_fn

@timer
def f(x):
    return x+1

```

pandas 性能提升

```python
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get


data = pd.DataFrame()
data['col1'] = np.random.normal(size = 1500000)
data['col2'] = np.random.normal(size = 1500000)

ddata = dd.from_pandas(data, npartitions=30)

import timeit


def myfunc(x,y): return y*(x**2+1)
def apply_myfunc_to_DF(df): return df.apply((lambda row: myfunc(*row)), axis=1)

def pandas_apply(): return apply_myfunc_to_DF(data)
def dask_apply(): return ddata.map_partitions(apply_myfunc_to_DF).compute(get=get)
def vectorized(): return myfunc(data['col1'], data['col2']  )

t_pds = timeit.Timer(lambda: pandas_apply())
print(t_pds.timeit(number=1))
# 52.0947

t_dsk = timeit.Timer(lambda: dask_apply())
print(t_dsk.timeit(number=1))
#19.0225

t_vec = timeit.Timer(lambda: vectorized())
print(t_vec.timeit(number=1))
# 0.025

import swifter
def apply_myfunc_to_DF_swiftly(df): return df.swifter.apply((lambda row: myfunc(*row)), axis=1)
def pandas_apply_swiftly(): return apply_myfunc_to_DF_swiftly(data)
t_swift = timeit.Timer(lambda: pandas_apply_swiftly())
print(t_swift.timeit(number=1))
# 74


def myfunc2(df):
    df['new'] = df.apply((lambda row: myfunc(*row)), axis=1)
    return df
import numpy as np
from multiprocessing import cpu_count,Pool
cores = cpu_count() #Number of CPU cores on your system
partitions = cores #Define as many partitions as you want
def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data
def apply_parall(): return parallelize(data,myfunc2)
t_parall = timeit.Timer(lambda: apply_parall())
print(t_parall.timeit(number=1))
# 18.73


```

可以看到，最快的还是向量化（在这个例子中，是 pandas 的 2000 倍）。所以，只要是能用向量化的地方，都尽量用向量化！
对不太好向量化的函数，用 dask 要比 pandas 快 3 倍; 手动并行化几乎和 dask 差不多的效率。
至于 swift, 看起来效果并不是很好

其他向量化的例子

```python

defdef  gt_5_bikes_vectorizedgt_5_bi (x):
    return np.where(x > 5, True, False)
```

# pythonic

python 3 中，合并字典的方式
x={'a':2,'b':4}
y={'a':5,'c':1}
z = {**x,**y}
z
{'a': 5, 'b': 4, 'c': 1}
注意，相同键，值用的是后一个。

# 线性回归

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression



model2 = LinearRegression()
model2.fit(x.wdd.values.reshape(27,1), x.n.values.reshape(27,))

matplotlib.rcParams['figure.figsize'] = [6,4]
plt.scatter(x.wdd.values, x.n.values,color='g')

plt.plot(x.wdd.values.reshape(27,1), model2.predict(x.wdd.values.reshape(27,1)),color='k')

plt.title(u"处理逃单量和逃单万订单之间的关系")

plt.xlabel(u'逃单万订单')
plt.ylabel(u'处理逃单门店量')
plt.show()

```

## python 环境管理

Pipfile 与 Pipfile.lock 是社区拟定的依赖管理文件，用于替代过于简陋的 requirements.txt

pipenv: 结合了 Pipfile 、pip 和 virtualenv

conda 本身也可以创建

```shell
conda create -n tf python=3.6 anaconda

# 安装包既可以用 conda，可以激活环境之后用pip3安装

# conda install -n tf -c https://conda.binstar.org/menpo opencv
# pip3 install tensorflow==2.1
# conda install -n tf pytorch torchvision -c pytorch
# python3 -m pip install mxnet==1.4.1 gluonts

#  Pycharm 等 IDE 在执行的时候，可以设置将 Source Root 加入到 PYTHONPATH，但如果是在 Shell 等运行，则我们的一些本地代码可能找不到
# 例如，我们执行 paradise 库下的一些代码，就会抱怨说找不到  paradise
# 这可以用如下方式解决： /Users/zhangxisheng/anaconda/envs/tf/lib/python3.6/site-packages/ 下创建一个 paradise.pth 文件，里面下上路径  `/Users/zhangxisheng/mtdp/paradise/` ,这样就可以了。


source activate tf

# 我们在这个环境下安装了 tensorflow 最新版本,以及 simple tensorflow serving
/Users/zhangxisheng/anaconda/envs/tf/bin/tensorboard --logdir=''



source deactivate

# 删除环境！
conda remove -n yourenvname --all

```

## 高级语法

1. jupyter 中，shift tab 一般是回退一个 tab，但是如果光标在函数上，会显示函数的注释

2. python3 中的 f-string

高级拆包

```python
a, *b = 1,2,3

# a
# 1
# b
# [2, 3]


a,*b,c = 1,2,3,4,5,6


# in 代替 or

if x == 1 or x == 2 or x == 3:
     pass

if x in (1,2,3):
     pass



```

## 常用

### 启动脚本个性化

import site
site.getusersitepackages()

eg. /Users/zhangxisheng/.local/lib/python2.7/site-packages

然后建立 sitecustomized.py 脚本，里面的语句将在 python 启动的时候自动执行。

但是，sitecustomized.py 脚本中 import 的包，还是不能找到。

应该可以 用 PYTHONSTARTUP 变量指向，但是确实这样是不好的做法，所以还是放弃自动引入包。

### PEP8

有时候抱怨 `module level import not at top of file`, 可以注释 # noqa, 就不会被检查了

## 疑难点

df.to_csv('foo.txt',index=False,header=False, quoting=csv.QUOTE_NONE)

```python
lst = [1, 2, 3, 4, 5, 6, 7, 8]
for x in lst:
    if x < 6:
        lst.remove(x)

lst
# [2, 4, 6, 7, 8]
```

因为迭代器这里是按照索引去得带的，当把 1 删除之后，2 其实成了列表 lst 的首个元素，但是迭代器还是取看第二个位置的，因此 2 被跳过了。

正确的写法

```python
[x for x in lst if x >= 6]
```

Another solution would be to iterate in reverse. That way, no elements can be skipped since removing an item from the list will only affect the indexes of elements that were already handled. This can either be done manually using index-based iteration starting from the end, or using reversed():

```python
lst = [1, 2, 3, 4, 5, 6, 7, 8]
for x in reversed(lst):
    if x < 6:
        lst.remove(x)
```

Finally, one can also iterate over a copy of the list, so when removing elements from the original list, the iterator is not affected.

<div class="c-callout c-callout–warning">
<strong class="c-callout__title">Warning</strong>
<p class="c-callout__paragraph">
不要在循环中用 remove 删除元素列表！
</p>
</div>

```python
for i in range(1, 4):
    print(i)
    break
else: # Not executed as there is a break
    print("No Break")
```

foo 循环后直接跟 else 的用法可还行？

### 函数的默认参数

> Default parameter values are always evaluated when, and only when, the “def” statement they belong to is executed

函数定义中绑定默认参数，而不是在函数执行时
合理性： 函数也是对象，在定义时被执行得到的对象

**def** 是一个可执行语句！

## 多进程并行 Parallel Multiprocessing

GIL 是计算机程序设计语言解释器用于同步线程的一种机制，它使得任何时刻仅有一个线程在执行。即便在多核心处理器上，使用 GIL 的解释器也只允许同一时间执行一个线程。

因为 GIL 的存在，多线程看来是不可能的了，好在，还可以用 multiprocessing 开启多进程。

IO 密集型任务选择 multiprocessing.dummy，CPU 密集型任务选择 multiprocessing

<h3 style="color: inherit;line-height: inherit;margin-top: 1.6em;margin-bottom: 1.6em;font-weight: bold;border-bottom: 2px solid rgb(239, 112, 96);font-size: 1.3em;"><span style="font-size: inherit;line-height: inherit;display: inline-block;font-weight: normal;background: rgb(239, 112, 96);color: rgb(255, 255, 255);padding: 3px 10px 1px;border-top-right-radius: 3px;border-top-left-radius: 3px;margin-right: 3px;">第一种</span>
</h3>

```python
%%time

from joblib import Parallel, delayed
from multiprocessing import cpu_count

executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
tasks = (delayed(preds_11)(poiid) for poiid in df.poiid.unique())
results = executor(tasks)

```

<h3 style="color: inherit;line-height: inherit;margin-top: 1.6em;margin-bottom: 1.6em;font-weight: bold;border-bottom: 2px solid rgb(239, 112, 96);font-size: 1.3em;"><span style="font-size: inherit;line-height: inherit;display: inline-block;font-weight: normal;background: rgb(239, 112, 96);color: rgb(255, 255, 255);padding: 3px 10px 1px;border-top-right-radius: 3px;border-top-left-radius: 3px;margin-right: 3px;">第二种</span>
</h3>

```python
from multiprocessing import Pool
pool = Pool()
results = pool.map(preds_11, df.poiid.unique())
pool.close()
pool.join()
```

## 应用

### 替换目录下所有文件中的某个关键词

```python

import os, re

def change(s):

    match = re.match('.*{% asset_img (.*) %}', s)
    if match:
        url = match.group(1)
        print url
        return '![](' + url + ')\n'
    else:
        return s


root = "/Users/zhangxisheng/Documents/personal/grocery-store-of-lancezhange/section8/"

for dirpath, dirnames, filenames in os.walk(root):
    for filepath in filenames:
        file_name = os.path.join(dirpath, filepath)
        if file_name.endswith('.md'):
            print file_name
            with open(file_name, 'r') as f:
                lines = f.readlines()
            with open(file_name, 'w') as f:
                for line in lines:
                    line = change(line)
                    f.write(line)

```

cpython
.pyx 文件

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.

series.dtype == np.float64:
fi16 = np.finfo(np.float16)

series.dtype == np.int64: Machine limits for integer types.
ii8 = np.iinfo(np.int8)

ntol = npos + nneg

## python3.8

## 参考资料

- [stackoverflow 上一些 python 问题合集 ](http://pyzh.readthedocs.org/en/latest/python-questions-on-stackoverflow.html)
- [python tips/intermediate python](http://book.pythontips.com/en/latest/index.html)

- [关于 python 的面试题](https://github.com/taizilongxu/interview_python)

* [The Best of the Best Practices(BOBP) GUide for Python](https://gist.github.com/sloria/7001839)

* [python-basics-numpy-regex](https://github.com/shik3519/machine-learning/blob/master/tutorials/003-python-basics-numpy-regex.ipynb)
