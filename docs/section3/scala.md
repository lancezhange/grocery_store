# Scala
开山鼻祖 Martin Odersky







### 语言特性

- case class: 无需用 new 实例化, 默认带有 getter 等定义，最为主要的：其实例可以用模式匹配

- Traits


### 设计模式之 scala 版本



### Scala VS. Java

参考

- [why should i learn scala](http://www.toptal.com/scala/why-should-i-learn-scala)

- [Java Vs. Scala: Divided We Fail](http://shipilev.net/blog/2014/java-scala-divided-we-fail/)

- [Java 8 vs Scala. The Difference in Approaches and Mutual Innovations](http://kukuruku.co/hub/scala/java-8-vs-scala-the-difference-in-approaches-and-mutual-innovations)

- [10 Differences Between Scala and Java 8: Part 1](http://www.codeproject.com/Articles/893211/Differences-Between-Scala-and-Java-Part)


Hello World in Java:

    public class HelloJava {
        public static void main(String[] args) {
            System.out.println("Hello World!");
        }
    }

Hello World in Scala:

    object HelloScala {
        def main(args: Array[String]): Unit = {
            println("Hello World!")
        }
    }

创建一个字符串列表

Java:

    List<String> list = new ArrayList<String>();
    list.add("1");
    list.add("2");
    list.add("3");

Scala:

    val list = List("1", "2", "3")


将字符串列表转化为整型列表
Java:

    List<Integer> ints = new ArrayList<Integer>();
    for (String s : list) {
        ints.add(Integer.parseInt(s));
    }

Scala:

    val ints = list.map(s => s.toInt)


创建传统的 java 对象
First, the Java version:

    public class User {
        private String name;
        private List<Order> orders;
        public User() {
            orders = new ArrayList<Order>();
        }
        public String getName() {
            return name;
        }
        public void setName(String name) {
            this.name = name;
        }
        public List<Order> getOrders() {
            return orders;
        }
        public void setOrders(List<Order> orders) {
            this.orders = orders;
        }
    }

    public class Order {
        private int id;
        private List<Product> products;
        public Order() {
            products = new ArrayList<Product>();
        }
        public int getId() {
            return id;
        }
        public void setId(int id) {
            this.id = id;
        }
        public List<Product> getProducts() {
            return products;
        }
        public void setProducts(List<Product> products) {
            this.products = products;
        }
    }

    public class Product {
        private int id;
        private String category;
        public int getId() {
            return id;
        }
        public void setId(int id) {
            this.id = id;
        }
        public String getCategory() {
            return category;
        }
        public void setCategory(String category) {
            this.category = category;
        }
    }


scala 版本

    class User {
        var name: String = _
        var orders: List[Order] = Nil
    }

    class Order {
        var id: Int = _
        var products: List[Product] = Nil
    }

    class Product {
        var id: Int = _
        var category: String = _
    }




总结一下，首先从形式上，scala 去除了语句结尾繁琐的分号，语法上更为灵活，例如函数名可以是任意符号(例如 `+++`)。

scala 中的类可以有参数，而 java 中的不行。

scala 没有 static, 而是将一切 static 都用单例对象取而代之
Java 中不支持运算符重载，而 scala 中可以。

scala 中，一切皆对象，包括数字、函数等都是对象，而 java 区分原型（如 int）和引用类型。


scala 中的无参方法和零参方法：前者无论定义还是调用，都没有括号，就好像字段一样


### [scala 宏](http://scalamacros.org/)
宏在代码生成等方面是很有用的

在2.10和2.11中都是实验性质，且只有　def 宏，后续会加入　Type Macros 等。

Macros are functions that are called by the compiler during compilation

```{scala}
def m(x: T): R = macro implRef
```

---

### scala 周边

- [Breeze](https://github.com/scalanlp/breeze)

    A numerical processing library for Scala

- [Akka](https://github.com/akka/akka)

- [sbt](https://github.com/sbt/sbt)


- Spark 大数据利器

- Play 大名鼎鼎的框架

- [Spray](https://github.com/spray/spray)

    另一款框架。A suite of scala libraries for building and consuming RESTful web services on top of Akka: lightweight, asynchronous, non-blocking, actor-based, testable

- logback 日志系统


更多参见 [Awesome scala](https://github.com/lauris/awesome-scala)

---

### 风格指南
[Databricks Scala Guide](https://github.com/databricks/scala-style-guide)


---

### 资料

- [A Scala Tutorial for Java Programmers](http://docs.scala-lang.org/tutorials/scala-for-java-programmers.html)

- [99 个 scala 编程问题](http://aperiodic.net/phil/scala/s-99/)

- [Scala and Design Patterns](http://www.scala-lang.org/old/sites/default/files/FrederikThesis.pdf)

- [Design Patterns in Scala](https://pavelfatin.com/design-patterns-in-scala/)

- [The Neophyte's Guide to Scala](http://danielwestheide.com/scala/neophytes.html)


