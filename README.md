GAN, 又称生成对抗网络, 也是 Generative Adversarial Nets 的简称

我们知道自然界的生物无时无刻都不在相互对抗进化，生物在往不被天敌发现的方向进化， 而天敌则是对抗生物的进化去更容易发现生物。 二者就是在做一个博弈。

GAN网络和生物对抗进化一样， 也是一个博弈的过程。 我们通过网络的Generator去生成图片， 通过Discriminator去判别图片是否是网络生成的还是真实图片。Generator 和 Discriminator相互对抗， Generator在往生成图片不被Discriminator发现为假的方向进化， 而Discriminator则是在往更准确发现图片为Generator生成的方向进化。 这是二者的一个博弈过程。


GAN_Mnist: 

loss：
	对于Discriminator： 我们max V(D,G)， 也就是最大化这个函数。 最大化log D(x)  ， 最大化log 1- D(G), 这表明我们在增强识别判断生成图片的能力，提高判断精确率。 相反， 这也是在促进Generator生成的图片更加真实

	对于Generato： 我们最小化log(1-D(G)),  也就是提高D(G), 即图片的真实度

改进：
	在框架中， 我们都是最小化loss, 所以对于max V(D,G),  我们可以min -V(D,G)
	对于Generator,  有论文表明使用 min -log(D(G))会更好

论文地址： https://arxiv.org/abs/1406.2661

代码结果：
	![image](https://github.com/yueqianlongma/GAN/raw/master/img/GAN_Mnist.png)




GAN_Anime:

1.使用转置卷积代替池化
2.使用Batch_Normalization在Generator和Discriminator， 但是Generator输出层不使用
3.移除卷积网络后的全连接层
4.Generator各个隐含层的激活函数使用relu, 输出层使用tanh
   Discriminator每一层都使用leaky_Relu, alpha=0.2
5.使用AdamOptimizer进行梯度下降， learning_rate=0.0002, beta1=0.5

DCGAN:论文地址：https://arxiv.org/abs/1511.06434

数据地址： 
	https://pan.baidu.com/s/14XPTghR1BsEa3K0oJPBhWw

代码结果：
	![image](https://github.com/yueqianlongma/GAN/raw/master/img/GAN_Anime.png)
