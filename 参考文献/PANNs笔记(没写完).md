# PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition   
>在计算机视觉和自然语言处理中，在大规模数据集上预训练的系统已经很好地推广到了几个任务中。然而，在用于音频模式识别的大规模数据集上对预训练系统的研究有限。在本文中，我们提出了在大规模AudioSet数据集上训练的预训练音频神经网络（PANN）。这些PANN被转移到其他与音频相关的任务。我们研究了由各种卷积神经网络建模的PANN的性能和计算复杂性。我们提出了一种称为Wavegram-Logmel-CNN的架构，使用log-mel频谱图和波形图作为输入特征。我们最好的PANN系统在AudioSet标签上实现了最先进的平均精度（mAP）0.439。

## 1. AUDIO TAGGING SYSTEMS  
### 1.1 CNNs  

CNN架构通常通过卷积层来提取特征，对于音频分类而言，通常使用`log-mel`频谱图作为输入。即将短时傅里叶变换`STFT`应用于时频波形图以计算频谱图，然后将`mel`滤波器组应用于频谱图，最后进行对数运算以提取`log-mel`频谱图

卷积神经网络架构如下图所示：

![2f3c1a1718b002fc62cb7e53bff58f81](https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/366cfee8-eb67-4dad-abb4-d3d5bf8cc8aa)

深度残差网络中的残差块可以提高模型收敛速度，解决网络过深导致的退化问题。

深度残差网络架构如下图所示：

![9f5f16af575094022368a16eb6542216](https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/b70bb279-c8e6-4441-8115-aa3a8f911041)

当模型在便携式设备实现时，计算复杂性是一个重要问题，MobileNets使用深度可分离卷积可以减少模型参数数量。其架构如下图所示：

![f46dd32a69d6625b7523c81b616f07c1](https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/630bd554-0f85-4978-ad02-d11c8e5d9c5a)

### 1.2 one-dimensional CNNs

上面的音频分类系统是基于`log-mel`频谱图，这是一种人工提取的特征，为了提高音频分类系统的性能，研究人员直接对时域波形图进行识别。

DaiNet将卷积核大小为80，步长为4的一维卷积作用于音频波形图进行特征提取；然后再使用卷积核大小为3，步长为4的意味卷积进行进一步特征提取；最后通过`softmax`层进行分类预测。其中每层一维卷积后跟着一个池化核大小为4的最大池化层。

`w`

