<!-- TOC -->

1. [Problems](#problems)

<!-- /TOC -->

# Problems

+ seqGAN
    + exector.py 
        + merge_data_for_disc 方法中的 decoder 方法中 使用 gen_model.step， 连续执行几次step 结果均不变
        + 可能是 参数问题导致迭代的太慢，又有可能是代码逻辑问题

+ beam search， antilm， pointer network
