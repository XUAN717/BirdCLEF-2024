# BirdCLEF2024 🐦
## 任务日志  

* 建立仓库，团队加入🎉

* 加入比赛  
  研究比赛规则、任务

* 研究比赛数据  
  完成train_metadata分析及可视化  
  完成音频数据可视化

* Kaggle运行BirdCLEF24: KerasCV Starter

* 合并pre分支，把参考文献文件夹合并过来了，感觉可以删掉pre，不用分支  
  添加了BirdCLEF24 KerasCV Starter的代码解释文件，作为研究参考  
  添加someidea.md，改进想法

* 5.19更新更新🀄  
  添加compare.md，2023年和已上传的对比

* 5.26        
  通过添加Dropout层,解决原模型存在的过拟合问题,详细更新在someidea.md          
  将mixup函数部分从全部进行改为以0.5的概率进行
                        
* 5.27                    
  加入了WandB部分可以实现共享文件，（具体可以看下你们是否可以看到我的版本）         
  加入之前几年的input以及对前几年数据进行处理               

* 6.3  
  effnet+fsr+cutmixup用以往数据做预训练24年数据训练，但存在问题（🚑️6.4问题解决）

* 6.4  
  forkbirdclef，0.65  
  还有段段那个也是0.65  
  （这个只用了24年的数据，可以往里面加以往数据做预训练？但是这样就得拆成两个notebook）
  effnet+fsr+cutmixup+pre-training用以往数据做预训练/2024数据训练

* 6.7  
  模型集成0.66  
  0.5+0.5
  
* 6.9🔥  
  0.6+0.4⚡️，0.7+0.3，0.8+0.2  

## 0 一些资料
### 0.1  GitHub Desktop

[【b站视频】]( https://www.bilibili.com/video/BV1o7411U7j6/?share_source=copy_web&vd_source=62d3967069c1f835b2792b2c6bc29ce3)

[【使用文档】](https://cnxfs.com.cn/download/GithubIntroductionForMembers.docx)

### 0.2  小玩意  
Gitm👽️ji  
https://gitmoji.dev/

Unicode 符号表  
https://symbl.cc/cn/unicode-table/#miscellaneous-symbols-and-pictographs

em🙂ji  
https://www.webfx.com/tools/emoji-cheat-sheet/

### 0.3 参考文献概览

预训练音频神经网络  
PANNs Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition

## 1 可视化  
Trainmeta分析与可视化  

[trainmeta_visualization.md](https://github.com/XUAN717/BirdCLEF-2024/blob/main/visualization/trainmeta_visualization.md)中部分结果  

<img src="https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/1f8729ab-c128-4c13-bc5e-55e903f0e9cb" width="500">

## 2 BirdCLEF24: KerasCV Starter
### 2.1 研究代码
code explanation.md  
得分0.60，暂以此作为baseline

### 2.2 算法改进
someidea.md  
compare.md

## 3 EffNet + FSR + CutMixUp  
### 3.1 预训练  
加入2020，2021，2022，2023年数据作为预训练，2024年数据训练（0.61）  
模型保存在https://www.kaggle.com/datasets/wengsilu/birdclef2024training

### 3.2.1 新模型  
BirdCLEF 2024:Species Identification from Audio（0.65）

### 3.2.2 加入新模型
加入新模型，分别预测结果，将预测结果取平均提交（0.66，97/917）

### 3.3 扩充数据
https://www.kaggle.com/datasets/wengsilu/allbirdclef2024

## 4 结果分析及优化
### 4.1 排名分析
<table>
    <tr>
        <td ><center><img src="https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/8502fc8d-8897-46bc-9d28-0983f312806b" width="500"></center></td>
        <td ><center><img src="https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/f237d880-9b1f-492c-82bf-b6dbbe43cfdc" width="500"></center></td>
    </tr>
</table>


## 项目开发流程
```mermaid
gantt
    title 项目开发流程
    todayMarker off
    section 探究
      创建环境           :done, 2024-04-22, 4d
      初步探究     :done, 2024-04-26, 7d
   
    section 实施
      初次提交      :done, 2024-04-28  , 16d
      优化算法      :done, 2024-05-14  , 18d
      测试提交      :done, 2024-04-28  , 44d
      结果分析优化  :2024-06-11  , 20d

    section 验收
      验收答辩    :2024-07-2, 5d
```

