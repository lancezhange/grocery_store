# 数据分析专题

抛开技术和理论，数据分析需要对数据的敏感度，以及提出关键问题并利用数据发现**洞见(insights)** 的能力。

例如，针对[ Pronto 城市公共自行车数据](https://www.prontocycleshare.com/datachallenge)（微软举办的数据比赛），你能提出哪些问题？[这里](http://nbviewer.jupyter.org/github/jakevdp/ProntoData/blob/master/ProntoData.ipynb)是一个范例。

类似还有[芝加哥城市公共自行车数据](https://www.divvybikes.com/datachallenge)（比赛已结束），[湾区城市公共自行车数据](http://www.bayareabikeshare.com/datachallenge-2014)（比赛已结束）， 仔细研读获胜者们的成果，你一定会深受启发。

我们知道，数据可视化对增进人们对数据的理解、捕捉数据中潜藏的模式，起着至关重要的作用，因此，掌握一些数据可视化的知识对数据分析从业者来说也是必要的。关于数据可视化，请参考[可视化专题](../section6/visualization.md)，这里不再赘述。

[Analyzing 1.1 Billion NYC Taxi and Uber Trips, with a Vengeance](http://toddwschneider.com/posts/analyzing-1-1-billion-nyc-taxi-and-uber-trips-with-a-vengeance/)

很多时候数据分析的目的在于讲一个好故事。

### 异常检测专题(Anomaly Detection)

异常检测的应用是显而易见的：反欺诈、等。

[the Numenta Anomaly Benchmark](https://github.com/numenta/NAB)

- [Anomaly Detection : A Survey](http://www.datascienceassn.org/sites/default/files/Anomaly%20Detection%20A%20Survey.pdf)

  综述文章

- [Outlier Analysis](http://charuaggarwal.net/outlierbook.pdf)

  一本专著

### 因果分析(Causal Analysis)

## 参考

- [Practical advice for analysis of large, complex data sets ](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html) by 谷歌搜索日志数据分析团队 2016 年的一篇博客文章

  1. Technical
     - Look at your distributions
     - Consider the outliers
     - Report noise/confidence
     - Look at examples
     - Slice your data 数据分组观察
     - Consider practical significance 考虑显著性
     - Check for consistency over time
  2. Process
     - Separate Validation, Description, and Evaluation
     - Confirm expt/data collection setup
     - Check vital signs
     - Standard first, custom second 先用标准化的度量
     - Measure twice, or more
     - Check for reproducibility
     - Check for consistency with past measurements
     - Make hypotheses and look for evidence
     - Exploratory analysis benefits from end to end iteration
  3. Social
     - Data analysis starts with questions, not data or a technique
     - Acknowledge and count your filtering
       - Acknowledge and clearly specify what filtering you are doing
       - Count how much is being filtered at each of your steps
     - Ratios should have clear numerator and denominators
     - Educate your consumers
     - Be both skeptic and champion
     - Share with peers first, external consumers second
     - Expect and accept ignorance and mistakes