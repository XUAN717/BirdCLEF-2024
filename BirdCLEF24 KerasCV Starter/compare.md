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

####################################################################################### 
BIRDCLEF = '24'
print(f"# BirdCLEF - 20{BIRDCLEF}")
tmp = df24.query("birdclef==@BIRDCLEF").sample(1)
#tmp.loc[:, 'filepath'] = tmp.filepath.str.replace(GCSPATH3, BASE_PATH3)
row = tmp.squeeze()
Display audio
display_audio(row)

这段代码涉及了一些操作，让我来逐步解释：

1.BIRDCLEF = '24': 这行代码将字符串 '24' 赋值给变量 BIRDCLEF。
2.print(f"# BirdCLEF - 20{BIRDCLEF}"): 这行代码打印了一个字符串，格式化了变量 BIRDCLEF 的值，生成了一个类似 # BirdCLEF - 2024 的输出。
3.tmp = df_24.query("birdclef==@BIRDCLEF").sample(1): 这行代码从 DataFrame df_24 中选取了符合条件 "birdclef==@BIRDCLEF" 的行，然后随机抽取了其中的一个样本，并将其存储在变量 tmp 中。
4.tmp.loc[:, 'filepath'] = tmp.filepath.str.replace(GCS_PATH3, BASE_PATH3): 这行代码对 tmp 中的 filepath 列进行了字符串替换操作，将列中的 GCS_PATH3 替换为 BASE_PATH3。这个操作似乎用于修改文件路径。我们并没有GCS_PATH3，注释掉，其中GCS_PATH是之前设备管理是作用的路径。
5.row = tmp.squeeze(): 这行代码将 DataFrame tmp 中的单行数据转换为 Series，并将其存储在变量 row 中。
6.display_audio(row): 这行代码调用了一个函数 display_audio()，并将 row 作为参数传递给该函数。根据函数名，它可能用于显示音频数据。

综上所述，这段代码的作用是打印出 BirdCLEF 的年份信息，然后从 DataFrame df_23 中选择符合条件的样本，进行路径替换操作，并最终显示音频数据。
之后按照不同的路径对其他年份的代码也进行分别的书写，不在一一列举。

###############分层交叉验证###############################         
    
from sklearn.model_selection import StratifiedKFold                                                                       
       
skf1 = StratifiedKFold(n_splits=25, shuffle=True, random_state=CFG.seed)                                                       
skf2 = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)                                                       

df_pre = df_pre.reset_index(drop=True)                                                                                     
df_24 = df_24.reset_index(drop=True)                                                                                                              

df_pre["fold"] = -1                                                                                                    
df_24["fold"] = -1                                                                                                                                


for fold, (train_idx, val_idx) in enumerate(skf1.split(df_pre, df_pre['primary_label'])):                           
    df_pre.loc[val_idx, 'fold'] = fold                                                                                
    

for fold, (train_idx, val_idx) in enumerate(skf2.split(df_24, df_24['primary_label'])):                                                         
    df_24.loc[val_idx, 'fold'] = fold                                                                                 
这段代码主要用于创建交叉验证（Cross Validation）的折（fold）。让我来逐步解释：                                                                       
                               
1.from sklearn.model_selection import StratifiedKFold: 导入了用于分层交叉验证的 StratifiedKFold 类。                                         
2.skf1 = StratifiedKFold(n_splits=25, shuffle=True, random_state=CFG.seed): 创建了一个 StratifiedKFold 对象 skf1，将数据分为 25 个折，设置了 shuffle=True 参数以随机打乱数据，random_state=CFG.seed 设置了随机种子。                                                          
3.skf2 = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed): 创建了另一个 StratifiedKFold 对象 skf2，将数据分为 CFG.num_fold 个折，同样设置了 shuffle=True 参数和 random_state=CFG.seed 设置了随机种子。                                                                        
4.df_pre = df_pre.reset_index(drop=True): 重置了 DataFrame df_pre 的索引，确保索引连续并从 0 开始。                                        
5.df_24 = df_24.reset_index(drop=True): 重置了 DataFrame df_24 的索引，同样确保索引连续并从 0 开始。                                            
6.为 DataFrame 添加了一个名为 'fold' 的新列，并将其初始化为 -1。                                                 
7.对于 BirdCLEF 数据集中的折分配（21、22 和 23），通过循环遍历 StratifiedKFold 对象 skf1 生成的分割器。每次迭代，使用 split() 方法获取训练集和验证集的索引，然后将验证集对应行的 'fold' 列更新为当前折数 fold。                                                              
8.对于 IBirdCLEF 数据集中的折分配（24），通过循环遍历 StratifiedKFold 对象 skf2 生成的分割器，操作与步骤 7 类似。                                      

这段代码的目的是将数据集中的每个样本分配到指定的交叉验证折中，并在 DataFrame 中添加一个 'fold' 列来表示每个样本所在的折。                                     

其中分类交叉验证的作用是什么：                                                                        
分层交叉验证的主要作用是确保在划分训练集和验证集时，每个类别在不同折中的分布相对均匀。这种类型的交叉验证适用于分类问题，特别是当样本分布不平衡或者存在多个类别时。                                                                           
以下是分层交叉验证的几个重要作用：                                         
1.保持类别分布的一致性： 在分层交叉验证中，每个折中的类别比例与整体数据集中的类别比例相似。这有助于确保在训练和验证过程中，每个类别都能够得到充分的代表性，避免某些类别在某些折中缺乏代表性的情况。                                                           
2.减少抽样偏差： 如果使用随机抽样进行交叉验证，存在可能在某些折中某些类别的样本数量过少或者过多的情况。分层交叉验证通过确保每个折中类别的均衡分布，减少了抽样偏差的影响，提高了模型评估的可靠性。                                                               
3.更好地评估模型的泛化性能： 由于分层交叉验证能够保持类别分布的一致性，因此在不同折中评估模型性能时，模型更有可能能够泛化到未见过的类别。这样可以更准确地评估模型的泛化性能，预测其在真实世界中的表现。                                                                

综上所述，分层交叉验证是一种有效的模型评估技术，特别适用于处理分类问题并且希望保持类别分布均衡的情况。                                  
#############################过滤以及上下采样#################################          
在之前给的基础版文件中已经有了过滤和上采样，因此在这里我们今天仅添加下采样。                def downsample_data(df, thr=500):                    
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
    这个函数是用于对数据进行下采样的操作。让我来解释其中的步骤：             

1.class_dist = df['primary_label'].value_counts(): 这行代码计算了每个类别的样本数量分布，其中 'primary_label' 是类别标签的列。                                
2.up_classes = class_dist[class_dist &gt; thr].index.tolist(): 这行代码选择了样本数量超过阈值 thr 的类别，即需要进行下采样的类别列表。                                  
3.down_dfs = []: 创建了一个空列表，用于存储下采样后的类别数据。                        
4.下面的循环用于处理每个需要进行下采样的类别：                                     


5.class_df = df.query("primary_label==@c"): 获取当前类别 c 的数据子集。               
6.df = df.query("primary_label!=@c"): 从原始数据中移除当前类别 c 的数据。               
7.class_df = class_df.sample(n=thr, replace=False, random_state=CFG.seed): 对当前类别 c 的数据进行随机抽样（无放回），使其样本数量等于阈值 thr。                    
8.down_dfs.append(class_df): 将下采样后的当前类别 c 的数据添加到列表 down_dfs 中。        


9.down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True): 将原始数据 df 与下采样后的数据列表 down_dfs 进行合并，并返回合并后的 DataFrame。                          

这个函数的作用是确保每个类别的样本数量不超过指定的阈值 thr，从而缓解数据集中类别不平衡的问题。       
####################对之前的数据训练####################################                
#Configurations                               
numclasses = CFG.numclasses2                     
df = df_pre.copy()                            
fold = 0  

1.num_classes = CFG.num_classes2:这一行将类的数量分配给变量num_classes。该值似乎是从名为CFG.num_classes2的配置对象或变量中检索的。                              
2.df = df_pre.copy():这一行创建了一个名为df_pre的DataFrame的副本，并将其赋值给一个新的变量df。这样做通常是为了防止在执行操作时修改原始DataFrame。                   
3.fold = 0:这一行将值0赋给变量fold。这个变量可能在以后的交叉验证设置中用于指示特定的折叠，或者将数据集划分为训练集和验证集。 

# Compute batch size and number of samples to drop                            
infer_bs = (CFG.batch_size*CFG.infer_bs)                         
drop_remainder = CFG.drop_remainder   

这部分代码似乎是在计算批大小，并确定在创建批时是否删除任何剩余的样本。这些计算有助于确保批次大小适当，以实现有效的训练和推理，并确保在整个过程中一致地处理数据。让我们来分析一下:          

1.infer_bs = (CFG.)batch_size * CFG.infer_bs):这一行通过将batch_size参数乘以infer_bs参数来计算推理批大小。看起来你正在扩展推理的批大小，可能是为了在推理期间适应比训练更大或更小的批大小。                                            
2.drop_remainder = CFG。drop_remainder:这一行分配CFG的值。到变量Drop_remainder。此参数可能控制是否在数据集批处理期间删除任何不适合完整批处理的剩余样本。如果drop_remainder设置为True，则所有剩余的样本都将被丢弃。如果设置为False，则它们将包含在最后一批中，即使它小于指定的批大小。                                                          

# Split dataset with cv filter                              
if CFG.cv_filter:                                 
    df = filter_data(df, thr=5)                               
    train_df = df.query("fold!=@fold | ~cv").reset_index(drop=True)         
    valid_df = df.query("fold==@fold & cv").reset_index(drop=True)            
else:                                    
    train_df = df.query("fold!=@fold").reset_index(drop=True)                    
    valid_df = df.query("fold==@fold").reset_index(drop=True)                       
这部分代码似乎将数据集分成训练集和验证集，可能会使用交叉验证过滤器。让我们来分析一下:

1.如果CFG。cv_filter:此条件检查是否启用了交叉验证过滤器。好像是CFG。Cv_filter是一个配置参数，控制是否应用交叉验证过滤器。                           
2.df = filter_data(df, thr=5):如果启用了交叉验证过滤器，这一行将使用5的阈值(由thr参数指定)对DataFrame df应用过滤器。filter_data函数可能会基于一些与交叉验证相关的标准来过滤DataFrame。                                                               
3.Train_df = df.query("fold!=@fold | ~cv").reset_index(drop=True):这一行通过查询df来创建训练DataFrame (train_df)，其中折叠不等于指定的折叠索引(fold!=@fold)或交叉验证列为False (~cv)的行。此操作有效地排除了属于验证折叠或已标记为交叉验证的样本。                  
4.Valid_df = df。查询(“= = @fold折叠,reset_index(drop=True):类似地，这一行通过查询df来创建验证DataFrame (valid_df)，查找折叠等于指定的折叠索引(fold==@fold)并且交叉验证列为True (cv)的行。此操作选择属于验证折叠并已标记为交叉验证的样本。                        
5.else::如果没有启用交叉验证过滤器(CFG. CFG. CFG)。cv_filter为False)，则执行此代码块。    
6.train_df = df.query("fold!=@fold").reset_index(drop=True):在这种情况下，训练DataFrame (train_df)是通过查询df中不等于指定的折叠索引(fold!=@fold)的行来创建的。这将排除属于验证折叠的样本。                                                         
7.valid_df = df.query("fold==@fold").reset_index(drop=True):验证DataFrame (valid_df)是通过查询df中折线等于指定折线索引(fold==@fold)的行来创建的。这将选择属于验证折叠的样本。     

总的来说，该代码段根据是否启用交叉验证过滤和指定的折叠索引，将数据集划分为训练集和验证集。 

# Upsample train data                                        
train_df = upsample_data(train_df, thr=50)                          
train_df = downsample_data(train_df, thr=500)                                 

# Get file paths and labels                                          
train_paths = train_df.filepath.values; train_labels = train_df.target.values         
valid_paths = valid_df.filepath.values; valid_labels = valid_df.target.values       

# Shuffle the file paths and labels                                         
index = np.arange(len(train_paths))                                          
np.random.shuffle(index)                            
train_paths  = train_paths[index]                                    
train_labels = train_labels[index]                                              

1.上采样和下采样:                                               
2.Upsampling: train_df = upsample_data(train_df, thr=50):这一行似乎是使用一个名为upsample_data的函数对训练数据进行上采样。阈值thr=50表明出现次数少于50次的样本被上采样。   
3.Downsampling: train_df = downsample_data(train_df, thr=500):这一行使用一个名为 
 downsample_data的函数对训练数据进行下采样。阈值thr=500表示出现超过500次的样本被下采样。   
4.文件路径和标签:                                     
5.Train_paths = train_df.filepath.values;Train_labels = train_df.target。values:从训练数据框(train_df)中提取文件路径和相应的标签。                                   
6.Valid_paths = valid_df.filepath.values;Valid_labels = valid_df.target。values:类似地，它从验证DataFrame (valid_df)中提取文件路径和标签。                              
7.洗牌:                                                    
8.index = np.arange(len(train_paths)):这将创建一个与训练样本数量相对应的索引数组。       
np.random.shuffle(index):随机打乱索引。                              
10.train_paths = train_paths[index]和train_labels = train_labels[index]:这两行按照打乱的索引同步打乱文件路径和标签，有效地打乱了训练数据。                            
总体而言，本节通过上采样和下采样调整类分布对数据进行预处理，并通过洗牌训练数据来保证随机性。                              
# For debugging                             
if CFG.debug:                                      
    min_samples = CFG.batch_size*CFG.replicas*2                             
    train_paths = train_paths[:min_samples]; train_labels = train_labels[:min_samples]
    valid_paths = valid_paths[:min_samples]; valid_labels = valid_labels[:min_samples]
    
# Ogg or Mp3                                     
train_ftype = list(map(lambda x: '.ogg' in x, train_paths))        
valid_ftype = list(map(lambda x: '.ogg' in x, valid_paths))              

# Compute the number of training and validation samples                      
num_train = len(train_paths); num_valid = len(valid_paths)                  

# Build the training and validation datasets                            
cache=False                                 
train_ds = build_dataset(train_paths, train_ftype, train_labels,                
                         batch_size=CFG.batch_size*CFG.replicas, cache=cache,  shuffle=True,                     
                        drop_remainder=drop_remainder, num_classes=num_classes) 
valid_ds = build_dataset(valid_paths, valid_ftype, valid_labels,      
                         batch_size=CFG.batch_size*CFG.replicas, cache=True,  shuffle=False,                    
                         augment=False, repeat=False, drop_remainder=drop_remainder,
                         take_first=True, num_classes=num_classes)           
这段代码似乎处理了额外的预处理步骤和数据集构造，以及一些调试配置。让我们来分析一下:

1.调试配置:                         
2.if CFG.debug::此条件检查是否开启了调试模式。                 
3.min_samples = CFG.batch_size*CFG。replicas*2:它计算调试所需的最小样本数，可能是为了限制数据集大小以实现更快的调试。                             
4.截断训练和验证数据:                               
5.训练路径=训练路径[:min_samples];train_labels = train_labels[:min_samples]:这将训练数据截断到指定的最小样本数。                                  
6.Valid_paths = Valid_paths [:min_samples];valid_labels = valid_labels[:min_samples]:类似地，它将验证数据截断为指定的最小样本数。                           
7.文件类型识别:                                          
8.Train_ftype = list(map(lambda x: ';ogg' in x, train_paths)):这一行创建一个布尔列表，指示每个训练文件是否具有.ogg扩展名。           
9.Valid_ftype = list(map(lambda x: ';ogg' in x, valid_paths)):类似地，它为验证文件创建一个布尔列表。               
10.计算样本数:                                      
11.num_train = len(train_paths):计算训练样本的数量。                     
12.num_valid = len(valid_paths):它计算验证样本的数量。                           
13.数据结构:                                    
14.train_ds = build_dataset(…):这一行使用build_dataset函数构造训练数据集。它传递必要的参数，如文件路径、文件类型、标签、批处理大小、缓存选项、shuffle选项和类数量。       
15.valid_ds = build_dataset(…):类似地，它构造验证数据集。          
build_dataset函数可能通过加载和预处理数据来准备训练数据集，例如解码音频文件、应用数据增强(如果启用)和批处理数据。                                       
总的来说，本节设置了用于训练和验证的数据集，并提供了限制数据集大小的可选调试配置。如果您有任何进一步的问题或需要更多的解释，请随时提问! 

# Print information about the fold and training       
print('#'*25); print('#### Pre-Training')           
print('#### Image Size: (%i, %i) | Model: %s | Batch Size: %i | Scheduler: %s'%
      (*CFG.img_size, CFG.model_name, CFG.batch_size*CFG.replicas, CFG.scheduler))   
print('#### Num Train: {:,} | Num Valid: {:,}'.format(len(train_paths), len(valid_paths)))                    

# Clear the session and build the model                          
K.clear_session()                                 
with strategy.scope():                                    
    model = build_model(CFG, model_name=CFG.model_name, num_classes=num_classes)    

print('#'*25)                                  

# Checkpoint Callback                                  
ckpt_cb = tf.keras.callbacks.ModelCheckpoint( 
    'birdclef_pretrained_ckpt.h5', monitor='val_auc', verbose=0, save_best_only=True,
    save_weights_only=False, mode='max', save_freq='epoch')             
# LR Scheduler Callback                                
lr_cb = get_lr_callback(CFG.batch_size*CFG.replicas)             
callbacks = [ckpt_cb, lr_cb]                                
这部分代码提供了关于折叠和训练设置的信息，在分布式策略范围内初始化模型，并设置检查点和学习率调度器回调。让我们来分析一下:                      

1.打印培训信息:                               
2.print('#'*25):打印一系列'#'字符以在视觉上分隔部分。                   
3.print('#### Pre-Training'):此标题表示预训练部分的开始。                     
4.打印配置详细信息:                 
5.图像大小、模型名称、批大小和调度器:这一行打印有关图像大小、模型名称、批大小和用于训练的调度器的信息。                             
6.Number of Training and Validation Samples:打印训练和验证样本的数量。    
7.模型初始化:                              
8.K.clear_session():清除当前Keras会话以释放内存并避免以前模型的混乱。         
9.使用strategy.scope():…:此代码块在分布式训练策略(strategy)的范围内初始化模型，这允许跨多个加速器(例如，gpu或tpu)进行有效的分布式训练。                            
10.model = build_model(…):通过使用指定的配置(CFG)和模型名称调用build_model函数来初始化模型。传递num_classes参数以确定用于分类的输出类的数量。              
11.回调设置:                   
12.ckpt_cb:这将初始化一个ModelCheckpoint回调，以保存基于验证AUC (val_auc)的最佳模型权重。它指定文件名、监控指标和保存频率。                           
13.lr_cb:它基于指定的批大小初始化学习率调度器回调。                        
14.callbacks = [ckpt_cb, lr_cb]:这将创建一个包含检查点和学习率调度器回调的回调列表。   
总的来说，本节设置了训练环境，包括打印配置细节，在分布式训练范围内初始化模型，以及为模型检查点和学习率调度设置回调。                                
# Training                          
history = model.fit(
    train_ds,                         
    epochs=2 if CFG.debug else CFG.epochs,                  
    callbacks=callbacks,                      
    steps_per_epoch=len(train_paths)/CFG.batch_size//CFG.replicas,       
    validation_data=valid_ds,                   
    verbose=CFG.verbose,                          
)                         

# Show training plot                  
if CFG.training_plot:                             
    plot_history(history)                       
这部分代码执行训练过程，并选择性地显示训练图。让我们来分析一下:

1.模型训练:                          
2.history = model.fit(…):这一行开始模型的训练。                      
3.参数:                          
4.train_ds:训练数据集。                             
5.epochs:训练的epoch数，如果CFG.debug else CFG.epochs则由2决定。如果启用了调试(CFG.debug为True)，则只使用2个epoch;否则为CFG中指定的epoch数。Epochs是用的。          
6.回调:训练期间应用的回调列表，包括检查点和学习率调度。                       
7.step_per_epoch:在声明一个epoch结束之前从数据集产生的步数(批)。它被计算为 
  len(train_paths)/CFG.batch_size//CFG.replicas。           
8.validation_data:验证数据集。                         
9.verbose:冗余模式(0、1或2)，控制训练期间打印的信息量。             
10.展示训练情节:                         
11.如果CFG。training_plot:此条件检查是否启用了显示训练图的选项。            
12.plot_history(history):如果启用，则调用此函数以显示训练历史图，该图通常显示诸如epoch的损失和准确性之类的指标。                                         
总的来说，本节利用提供的训练和验证数据集，并通过回调监控其进度，编排模型的训练过程。此外，如果需要，它还提供了通过绘图将训练历史可视化的选项。              

###################音频以及频谱增强############################################################3                    
其作用有：                                  
1.提高模型的鲁棒性： 增强可以在训练数据中引入多样性，使模型更具鲁棒性。通过引入不同类型的噪声、畸变或变换，模型可以学习对各种情况进行适应，从而在实际应用中表现更好。                                                             
2.减轻过拟合： 通过在训练数据上引入随机变换，增强技术有助于减轻模型对训练数据的过度拟合。这使得模型更能够泛化到未见过的数据。                    
3.增加数据样本量： 增强技术可以通过生成经过变换的数据样本来增加训练数据的数量。这对于在数据量有限的情况下训练深度学习模型特别有用，可以提高模型的性能。         
4.提高模型的性能： 在某些情况下，增强技术可以改善模型的性能。例如，在语音识别任务中，引入环境噪声或其他变换可以使模型更好地适应不同的录音环境。          
5.增强特征表示： 对音频数据进行增强可以帮助模型学习更丰富和鲁棒的特征表示。例如，通过添加随机噪声或变换，可以帮助模型更好地理解语音中的语义内容。        
6.对抗对抗性攻击： 在某些情况下，增强技术可以提高模型对对抗性攻击的鲁棒性。通过对输入数据进行随机变换，可以使攻击者更难以找到有效的对抗样本。          
总的来说，音频和频谱图增强技术可以帮助改善模型的训练效果，增强模型的鲁棒性，并提高模型在实际应用中的性能。                                             
