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



////////////////////////////////////////////////////////////////////////////////////////
增加新内容
1.WandB
# Import wandb library for logging and tracking experiments
import wandb

# Try to get the API key from Kaggle secrets
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
这段代码使用了 WandB 库，用于记录和跟踪实验。让我来解释一下代码的功能：

1.import wandb: 导入 WandB 库，用于实验记录和跟踪。
2.from kaggle_secrets import UserSecretsClient: 从 Kaggle Secrets 模块中导入 UserSecretsClient 类，用于从 Kaggle Secrets 中获取 API 密钥。
3.user_secrets = UserSecretsClient(): 创建一个 UserSecretsClient 对象，用于访问 Kaggle Secrets。
4.api_key = user_secrets.get_secret("WANDB"): 从 Kaggle Secrets 中获取名为 "WANDB" 的 API 密钥。
5.wandb.login(key=api_key): 使用获取到的 API 密钥登录到 WandB，以便记录实验。
6.anonymous = None: 设置匿名模式为 None，表示使用获取到的 API 密钥登录。
7.except: 如果无法从 Kaggle Secrets 中获取 API 密钥，则执行以下操作：
a. anonymous = 'must': 将匿名模式设置为 'must'，表示必须以匿名方式登录。
b. wandb.login(anonymous=anonymous, relogin=True): 以匿名模式重新登录到 WandB，以便记录实验。

这段代码的作用是尝试从 Kaggle Secrets 中获取 WandB 的 API 密钥，如果成功则使用该密钥登录到 WandB，否则以匿名方式登录。这样可以确保在 Kaggle 上运行实验时能够记录和跟踪实验结果。
作用：
确保在 Kaggle 上运行实验时可以跟踪记录实验结果有几个重要的原因：

1.实验复现和共享： 在 Kaggle 上进行的实验通常是与机器学习和数据科学相关的项目。记录实验结果可以帮助其他人复现你的实验，并且可以共享给其他人，从而促进知识的传播和共享。
2.实验比较和评估： 通过记录实验结果，你可以比较不同模型、算法或参数设置的性能，并对它们进行评估。这有助于确定哪些方法在特定任务上效果最好，并为进一步的改进提供参考。
3.进度追踪和调试： 记录实验结果可以帮助你跟踪项目的进展，并对模型性能的变化进行调试和分析。如果某个模型在某个阶段性能下降，你可以通过比较实验记录找出原因并进行调整。
4.实验复现和验证： 实验结果记录可以帮助你验证实验的有效性和稳健性。通过记录实验过程和结果，你可以更容易地验证你的方法是否是可靠的，并且可以重复实验以确认结果的一致性。
5.团队协作： 如果你是在团队中工作，记录实验结果可以促进团队成员之间的合作和沟通。团队成员可以查看实验记录，了解项目的当前状态，并为项目的进展提供反馈和建议。

总之，记录实验结果是进行科学研究和工程实践中的重要步骤，可以帮助你理解、评估和改进你的模型和方法。在 Kaggle 这样的平台上进行实验时，确保能够跟踪记录实验结果，有助于提高项目的可重复性、可比较性和合作性。

2.对之前的数据也进行训练
以下为定义路径
BASE_PATH0 = '/kaggle/input/birdsong-recognition'
BASE_PATH1 = '/kaggle/input/birdclef-2021'
BASE_PATH2 = '/kaggle/input/birdclef-2022'
BASE_PATH3 = '/kaggle/input/birdclef-2023'
BASE_PATH4 = '/kaggle/input/birdclef-2024'
BASE_PATH5 = '/kaggle/input/xeno-canto-bird-recordings-extended-a-m'
BASE_PATH6 = '/kaggle/input/xeno-canto-bird-recordings-extended-n-z'
​
BirdCLEF - 24
df_24 = pd.read_csv(f'{BASE_PATH4}/train_metadata.csv')
df_24['filepath'] = BASE_PATH4 + '/train_audio/' + df_24.filename
df_24['target'] = df_24.primary_label.map(CFG.name2label)
df_24['birdclef'] = '24'
df_24['filename'] = df_24.filepath.map(lambda x: x.split('/')[-1])
df_24['xc_id'] = df_24.filepath.map(lambda x: x.split('/')[-1].split('.')[0])

# Display rwos
print("# Samples in BirdCLEF 24: {:,}".format(len(df_24)))
df_24.head(2).style.set_caption("BirdCLEF - 24").set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'blue'),
        ('font-size', '16px')
    ]
}])
（以下为23年的解释示例，不过没有CFG定义可以之后一起讨论）
这段代码涉及使用 Pandas 和 TensorFlow 处理数据，并在输出中显示 DataFrame 的前两行。

1.df_23 = pd.read_csv(f'{BASE_PATH3}/train_metadata.csv'): 这一行使用 Pandas 的 read_csv 函数从指定路径读取 CSV 文件，并将其存储在名为 df_23 的 DataFrame 中。
2.df_23['filepath'] = GCS_PATH3 + '/train_audio/' + df_23.filename: 这一行创建一个新列 'filepath'，其值为 GCS 路径加上 'train_audio/' 和 DataFrame 中 'filename' 列的值。
3.df_23['target'] = df_23.primary_label.map(CFG.name2label): 这一行根据 'primary_label' 列的值映射到 CFG.name2label 字典中的对应值，并将结果存储在新的 'target' 列中。
4.df_23['birdclef'] = '23': 这一行创建一个名为 'birdclef' 的新列，并将所有行的值设置为 '23'。
5.df_23['filename'] = df_23.filepath.map(lambda x: x.split('/')[-1]): 这一行根据 'filepath' 列的值，使用 split 函数取得文件名，并将结果存储在新的 'filename' 列中。
6.df_23['xc_id'] = df_23.filepath.map(lambda x: x.split('/')[-1].split('.')[0]): 这一行根据 'filepath' 列的值，使用 split 函数取得文件名，并再次使用 split 函数取得文件名中的 ID 部分，并将结果存储在新的 'xc_id' 列中。
7.assert tf.io.gfile.exists(df_23.filepath.iloc[0]): 这一行使用 TensorFlow 的 tf.io.gfile.exists 函数检查 DataFrame 中第一行文件路径是否存在。如果不存在，将会引发 AssertionError。
8.print("# Samples in BirdCLEF 23: {:,}".format(len(df_23))): 这一行输出 BirdCLEF 23 数据集中样本的数量。
9.最后一行使用 Pandas 的样式设置方法将 DataFrame 的前两行显示为带标题和样式的格式，并使用 set_caption 和 set_table_styles 设置标题和表格样式。

这段代码的目的是加载数据、处理文件路径和相关信息，并以漂亮的格式显示 DataFrame 的前两行。
# Xeno-Canto Extend by @vopani
df_xam = pd.read_csv(f'{BASE_PATH5}/train_extended.csv')
df_xam['filepath'] = BASE_PATH5 + '/A-M/' + df_xam.ebird_code + '/' + df_xam.filename
df_xnz = pd.read_csv(f'{BASE_PATH6}/train_extended.csv')
df_xnz['filepath'] = BASE_PATH6 + '/N-Z/' + df_xnz.ebird_code + '/' + df_xnz.filename
df_xc = pd.concat([df_xam, df_xnz], axis=0, ignore_index=True)
df_xc['primary_label'] = df_xc['ebird_code']
df_xc['scientific_name'] = df_xc['sci_name']
df_xc['common_name'] = df_xc['species']
df_xc['target'] = df_xc.primary_label.map(CFG.name2label2)
df_xc['birdclef'] = 'xc'
assert tf.io.gfile.exists(df_xc.filepath.iloc[0])
​
# BirdCLEF-2021
df_21 = pd.read_csv(f'{BASE_PATH1}/train_metadata.csv')
df_21['filepath'] = BASE_PATH1 + '/train_short_audio/' + df_21.primary_label + '/' + df_21.filename
df_21['target'] = df_21.primary_label.map(CFG.name2label2)
df_21['birdclef'] = '21'
corrupt_paths = ['/kaggle/input/birdclef-2021/train_short_audio/houwre/XC590621.ogg',
                 '/kaggle/input/birdclef-2021/train_short_audio/cogdov/XC579430.ogg']
df_21 = df_21[~df_21.filepath.isin(corrupt_paths)] # remove all zero audios
assert tf.io.gfile.exists(df_21.filepath.iloc[0])
​
# BirdCLEF-2022
df_22 = pd.read_csv(f'{BASE_PATH2}/train_metadata.csv')
df_22['filepath'] = BASE_PATH2 + '/train_audio/' + df_22.filename
df_22['target'] = df_22.primary_label.map(CFG.name2label2)
df_22['birdclef'] = '22'
assert tf.io.gfile.exists(df_22.filepath.iloc[0])
​
# BirdCLEF-2023
df_23 = pd.read_csv(f'{BASE_PATH3}/train_metadata.csv')
df_23['filepath'] = BASE_PATH3 + '/train_audio/' + df_23.filename
df_23['target'] = df_23.primary_label.map(CFG.name2label2)
df_23['birdclef'] = '23'
assert tf.io.gfile.exists(df_23.filepath.iloc[0])
​
# Merge 2021 and 2022 for pretraining
df_pre = pd.concat([df_21,df_22,df_23,df_xc], axis=0, ignore_index=True)
df_pre['filename'] = df_pre.filepath.map(lambda x: x.split('/')[-1])
df_pre['xc_id'] = df_pre.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
nodup_idx = df_pre[['xc_id','primary_label','author']].drop_duplicates().index
df_pre = df_pre.loc[nodup_idx].reset_index(drop=True)
​
# # Remove duplicates
df_pre = df_pre[~df_pre.xc_id.isin(df_24.xc_id)].reset_index(drop=True)
corrupt_mp3s = json.load(open('/kaggle/input/birdclef-corrupt-mp3-files-ds/corrupt_mp3_files.json','r'))
df_pre = df_pre[~df_pre.filepath.isin(corrupt_mp3s)]
df_pre = df_pre[['filename','filepath','primary_label','secondary_labels',
                 'rating','author','file_type','xc_id','scientific_name',
                'common_name','target','birdclef','bird_seen']]
# Display rows
print("# Samples for Pre-Training: {:,}".format(len(df_pre)))
df_pre.head(2).style.set_caption("Pre-Training Data").set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'blue'),
        ('font-size', '16px')
    ]
}])
​
# Show distribution
plt.figure(figsize=(8, 4))
df_pre.birdclef.value_counts().plot.bar(color=[cmap(0.0),cmap(0.25), cmap(0.65), cmap(0.9)])
plt.xlabel("Dataset")
plt.ylabel("Count")
plt.title("Dataset distribution for Pre-Training")
plt.show()

1.corrupt_mp3s = json.load(open('/kaggle/input/birdclef-corrupt-mp3-files-ds/corrupt_mp3_files.json','r')): 这一行加载了一个 JSON 文件，其中包含了一些损坏的音频文件的文件路径列表。

2.df_pre = df_pre[~df_pre.filepath.isin(corrupt_mp3s)]: 这行代码移除了 df_pre 中包含在 corrupt_mp3s 列表中的损坏音频文件的样本。​

这段代码看起来是用于预处理预训练数据集的一部分。让我们逐行解释它：

3.df_pre = df_pre[~df_pre.xc_id.isin(df_24.xc_id)].reset_index(drop=True): 这行代码从预训练数据集 df_pre 中移除了与 BirdCLEF-2024 数据集 (df_24) 中重复的样本，通过检查它们的 xc_id 列是否存在于 df_24 中。之后，通过 reset_index(drop=True) 重新设置了索引。

4.corrupt_mp3s = json.load(open('/kaggle/input/birdclef-corrupt-mp3-files-ds/corrupt_mp3_files.json','r')): 这行代码加载了一个 JSON 文件，其中包含了一些损坏的 MP3 文件的文件路径列表。

5.df_pre = df_pre[~df_pre.filepath.isin(corrupt_mp3s)]: 这行代码从 df_pre 中移除了与损坏 MP3 文件列表 corrupt_mp3s 中的文件路径匹配的样本。

6.df_pre = df_pre[['filename','filepath','primary_label','secondary_labels', 'rating','author','file_type','xc_id','scientific_name', 'common_name','target','birdclef','bird_seen']]: 这行代码选择了一些特定的列，以便于后续的数据展示和分析。

7.print("# Samples for Pre-Training: {:,}".format(len(df_pre))): 这行代码打印了经过预处理后的预训练数据集的样本数量。

8.df_pre.head(2).style.set_caption("Pre-Training Data").set_table_styles([...]): 这行代码展示了预处理后的预训练数据集的前两行样本，并设置了标题和表格样式。

综上所述，这段代码的作用是从预训练数据集中移除了与 BirdCLEF-2024 数据集中重复的样本以及损坏的 MP3 文件，并选择了一些特定的列进行展示。最后，打印了预处理后的数据集样本数量，并展示了前两行样本。
