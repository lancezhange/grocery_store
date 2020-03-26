# Bert

[[TOC]]

</br>
<section style="margin-bottom:-16px">
<section style="margin-top:0px;margin-right:0px;margin-bottom:0px;margin-left:2em;padding-top:2px;padding-right:1em;padding-bottom:2px;padding-left:1em;max-width:100%;display:inline-block;background-image:none;background-color:rgb(196, 212, 218);color:rgb(61, 88, 98);font-size:16px;text-align:center;letter-spacing:1.5px;line-height:1.75em;border-top-left-radius:16px;border-top-right-radius:16px;border-bottom-right-radius:16px;border-bottom-left-radius:16px;box-sizing:border-box;word-wrap:break-word;">
<strong>前言</strong>
</section>
</section>
<section style="margin-top:0px;margin-right:0px;margin-bottom:20px;margin-left:0px;padding-top:2.5em;padding-right:1em;padding-bottom:1em;padding-left:1em;max-width:100%;box-sizing:border-box;border-top-width:1px;border-right-width:1px;border-bottom-width:1px;border-left-width:1px;border-top-style:solid;border-right-style:solid;border-bottom-style:solid;border-left-style:solid;border-top-color:rgb(196, 212, 218);border-right-color:rgb(196, 212, 218);border-bottom-color:rgb(196, 212, 218);border-left-color:rgb(196, 212, 218);border-top-left-radius:10px;border-top-right-radius:10px;border-bottom-right-radius:10px;border-bottom-left-radius:10px;word-wrap:break-word;">Bert 在 NLP 发展历史上是一个划时代的存在，因此，这里将 Bert 从 NLP 一章中单拿出来学习。
</section>
</br>

## 模型迭代

### Bert

### ALBert

A lite version of Bert.

主要是通过矩阵分解和跨层参数共享来做到对参数量的减少，除此以外也是用 **SOP（Sentence Order Prediction）** 替换了 **NSP（Next Sentence Prediction）**，并且停用了 Dropout，最后 GLUE 上一举达到了 SOTA 的效果

### RoBERTa

### Tinybert

### Reformer

使用局部敏感哈希的点乘注意力将 transorfmer 中的 attention 复杂度从 O(N^2 ) 变为 O(N log N)，将算法复杂度降低，得到了更高的效率。
