2023年和已上传的对比
2023文件来源：https://www.kaggle.com/code/awsaf49/birdclef23-pretraining-is-all-you-need-train/notebook - BirdCLEF-2023-%F0%9F%90%A6
2024github的公共文件

Class CFG:（部分优化）
# Input image size and batch size
    upsample_thr = 50 # min sample of each class (upsample)
    cv_filter = True # always keeps low sample data in train
upsample_thr 设置每个类别的最小样本
cv_filter 始终保持低样本在训练中（降低模型对有标签的样本的依赖？）

# STFT parameters
normalize = True
normalize = True使结果在[0,1]之间显示，最后的处理结果与均值滤波相同

# Inference batch size, test time augmentation, and drop remainder推理批大小、测试时间增加和删除余数
    infer_bs = 2
    tta = 1
    drop_remainder = True
drop remainder=True表示如果数据集最后个批次的样本数不足一个批次大小，就将其丢弃。这是因为在训练神经网络时，通常要求每个批次大小相等，这样才能进行并行计算。如果最后一个批次不足一个批次大小，那么就会导致无法进行并行计算，因此需要丢弃。

# Pretraining, neck features, and final activation function
    pretrain = 'imagenet'
    neck_features = 0
    final_act = 'softmax'
pretrain = 'imagenet' 预处理‘imagenet’数据集

# Learning rate, optimizer, and scheduler
    lr = 1e-3
    scheduler = 'cos'
    optimizer = 'Adam' # AdamW, Adam
scheduler = 'cos'学习率调整为余弦形式变化
optimizer = 'Adam' optimizer 参数指定了用于训练模型的优化器。Adam 是一种常见的优化器，它被认为在大多教情况下都是一个合适的选择。它使用自造应学习率Q的方法来调整模型的权重，从而使训练过程收敛到最优解。

# Loss function and label smoothing
    loss = 'CCE' # BCE, CCE
    label_smoothing = 0.05 # label smoothing
loss = 'CCE'用于多分类问题，衡量输出与真实标签的交叉熵。
label_smoothing 标签平滑防止过拟合

  # Time Freq masking
    freq_mask_prob=0.50
    num_freq_masks=1
    freq_mask_param=10
    time_mask_prob=0.50
    num_time_masks=2
    time_mask_param=25
本段代码中timemasking用于时间增强。（各项数据的设定没看明白）
https://blog.csdn.net/deephub/article/details/123704862?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171601891016800225516375%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171601891016800225516375&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-123704862-null-null.142^v100^pc_search_result_base2&utm_term=timemasking&spm=1018.2226.3001.4187

# Audio Augmentation Settings
音频增强设置

# Training Settings
    target_col = ['target']
    tab_cols = ['filename']
    monitor = 'auc'
auc为一个概率值，可以更好的分类

# Import wandb library for logging and tracking experiments 权重与偏差
import wandb

# Try to get the API key from Kaggle secrets获取秘钥
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("WANDB")
    # Login to wandb with the API key
    wandb.login(key=api_key)
    # Set anonymous mode to None
    anonymous = None
except:
    # If Kaggle secrets are not available, set anonymous mode to 'must'
    anonymous = 'must'
    # Login to wandb anonymously and relogin if needed
    wandb.login(anonymous=anonymous, relogin=True)
wandb 权重与偏差的工具（可以用来记录音频文件？），可以用来做错误分析。固定的。此段为wandb的强制登录

set up device自动检测硬件

Dataset path使用远端Tpu作为设备时的Gcs路径

BirdCLEF - 20, 21, 22 & Xeno-Canto Extend
# BirdCLEF-2020
# BirdCLEF-2021
# BirdCLEF-2022
此段函数侧重于查看之前三年的数据集 （可以参考）


def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

def downsample_data(df, thr=500):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df
Filter: As there is even only one sample for some classes we need to make sure they are in the train data using filtering. We can do this by always keeping them in the train data and do cross-validtion on the rest of the data.
过滤器:由于有些类甚至只有一个样本，我们需要使用过滤来确保它们在训练的数据中。我们可以通过始终将它们保存在训练数据中并对其余数据进行交叉验证来做到这一点。
def filter_data(df, thr=5):
计算每个类的样本数量，定义了thr=5,选择样本数量小于thr的，并且添加一个新列来选择交叉验证的样本，对于样本数小于thr的设置cv = false。返回过滤后的数据帧。

Upsample: Even in the filtered data there are some minority classes with very few samples. To amend the class imbalance we can try upsampling those classes. Following function will simply upsample the train data for minory class which has very few samples. This can potentially mitigate the classic "Long Tail" problem.
上样本:即使在过滤后的数据中，也有一些样本很少的少数类。为了修正类的不平衡，我们可以尝试对这些类进行上采样。下面的函数将简单地对样本很少的小班的训练数据进行上采样。这可以潜在地缓解经典的“长尾”问题。
def upsample_data(df, thr=20)
其中各个语句分别为以下含义：
1）获取类分布
2）识别样本数量小于阈值的类
3）创建一个空列表来存储上采样的数据框
4）循环遍历欠采样类并对它们进行上采样
在其中：
class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
在CFG.seed中随机选取n个数据填充
5）连接上采样的数据帧和原始数据帧

Downsample: Ensure maximum sample of a class.
确保类的最大样本（分析与up的相反）

Important
在2023年中提到了数据增强部分，我认为可以大大改进。其观点主要是认为传统的视觉增强功能不能对我们将音频作为图像训练起到较好的效果，因此增加了Audio Aug和SepcAug进行了音频数据和谱图数据的增强。

MelSpectrogram
该层将音频数据转换为GPU/TPU上的频谱数据。这样可以大大加快速度。
SpecAug - Time Frequency Masking
这一层在增强训练期间，掩盖了时间框架和频率范围？（没看懂）

此外Loss,Metric&Optmizer使用分类交叉熵(CCE)损失进行优化，同时使用AUC (PR曲线)和准确性作为性能指标。以及提高在GPU的运行速度。

