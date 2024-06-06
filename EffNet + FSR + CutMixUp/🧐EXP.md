# birdclef24_EffNet+FSR+CutMixUp[TRAIN]

* 2020, 2021, 2022 & 2023比赛数据和 Xeno-Canto数据作为预训练数据
* 2024比赛数据作为训练数据

-----------------
### 加入新模型   
分数0.66，97/917

-----------------
### 问题已解决
这是2023年pre-training的（公榜0.8分）  
<img src="https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/57cc075b-b60c-4fee-85db-1b01865238f5" width="600">

这是我的  
<img src="https://github.com/XUAN717/BirdCLEF-2024/assets/97745870/c29fb46e-cef5-4495-95ed-81e2c9f2bc3a" width="600">

-----------------
### pre-training存在错误如下

```
#########################
#### Pre-Training
#### Image Size: (128, 384) | Model: EfficientNetB1 | Batch Size: 32 | Scheduler: cos
#### Num Train: 95,611 | Num Valid: 3,686
#########################
2024-06-03 10:09:02.515017: E tensorflow/core/grappler/optimizers/meta_optimizer.cc:961] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inmodel/efficientnet-b1/block1b_drop/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
Cell In[60], line 78
     75 callbacks = [ckpt_cb, lr_cb]
     77 # Training
---> 78 history = model.fit(
     79     train_ds, 
     80     epochs=2 if CFG.debug else CFG.epochs, 
     81     callbacks=callbacks, 
     82     steps_per_epoch=len(train_paths)/CFG.batch_size//CFG.replicas,
     83     validation_data=valid_ds, 
     84     verbose=CFG.verbose,
     85 )
     87 # Show training plot
     88 if CFG.training_plot:

File /opt/conda/lib/python3.10/site-packages/keras/src/utils/traceback_utils.py:70, in filter_traceback.<locals>.error_handler(*args, **kwargs)
     67     filtered_tb = _process_traceback_frames(e.__traceback__)
     68     # To get the full stack trace, call:
     69     # `tf.debugging.disable_traceback_filtering()`
---> 70     raise e.with_traceback(filtered_tb) from None
     71 finally:
     72     del filtered_tb

File /opt/conda/lib/python3.10/site-packages/tensorflow/python/eager/execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     51 try:
     52   ctx.ensure_initialized()
---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
     54                                       inputs, attrs, num_outputs)
     55 except core._NotOkStatusException as e:
     56   if name is not None:

InvalidArgumentError: Graph execution error:

2 root error(s) found.
  (0) INVALID_ARGUMENT:  read 672768 from 0 failed: 0
	 [[{{node cond/IO>AudioDecodeMP3}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]]
	 [[IteratorGetNext]]
	 [[div_no_nan/ReadVariableOp_1/_280]]
  (1) INVALID_ARGUMENT:  read 672768 from 0 failed: 0
	 [[{{node cond/IO>AudioDecodeMP3}}]]
	 [[MultiDeviceIteratorGetNextFromShard]]
	 [[RemoteCall]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_function_117994]
```
### ChatGPT   
这个错误表明在处理音频文件时发生了问题，尤其是在 AudioDecodeMP3 操作中读取数据时出现了无效的参数。这个问题通常与数据输入管道中的音频文件格式或文件路径有关。以下是一些可能的解决方法和调试步骤：

### 评论
评论中也有人更换数据出现了同样的问题



