**DE⫶TR**: 使用 Transformer 的端到端目标检测
========

[![支持乌克兰](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)

这是 **DETR** (**DE**tection **TR**ansformer) 的 PyTorch 训练代码和预训练模型。
我们用一个 Transformer 替代了完整复杂的手工设计的目标检测流程，使用 ResNet-50 作为骨干网络，在 COCO 数据集上获得了 **42 AP** 的性能，同时只使用了一半的计算量（FLOPs）和相同数量的参数。使用 PyTorch 只需 50 行代码即可完成推理。

![DETR](.github/DETR.png)

**这是什么**。与传统的计算机视觉技术不同，DETR 将目标检测视为一个直接的集合预测问题。它包含一个基于集合的全局损失函数，通过二分图匹配强制产生唯一的预测结果，以及一个 Transformer 编码器-解码器架构。
给定一组固定的小型学习对象查询，DETR 推理对象之间的关系和全局图像上下文，直接并行输出最终的预测集合。由于这种并行特性，DETR 非常快速和高效。

**关于代码**。我们相信目标检测不应该比分类更困难，也不应该需要复杂的库来进行训练和推理。
DETR 的实现和实验非常简单，我们提供了一个[独立的 Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)，展示如何只用几行 PyTorch 代码进行 DETR 推理。
训练代码遵循这一理念 - 它不是一个库，而只是一个导入模型和损失函数定义的[main.py](main.py)，使用标准的训练循环。

此外，我们在 d2/ 文件夹中提供了 Detectron2 包装器。更多信息请参见该文件夹中的 readme。

详情请参见 Nicolas Carion、Francisco Massa、Gabriel Synnaeve、Nicolas Usunier、Alexander Kirillov 和 Sergey Zagoruyko 的论文[End-to-End Object Detection with Transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers)。

请查看我们的[博客文章](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers/)了解更多关于使用 Transformer 进行端到端目标检测的信息。

# 模型库
我们提供了基线 DETR 和 DETR-DC5 模型，并计划在未来包含更多模型。
AP 是在 COCO 2017 val5k 上计算的，推理时间是在前 100 张 val5k COCO 图像上使用 torchscript transformer 测量的。

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名称</th>
      <th>骨干网络</th>
      <th>训练轮数</th>
      <th>推理时间</th>
      <th>框 AP</th>
      <th>下载链接</th>
      <th>大小</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>500</td>
      <td>0.036</td>
      <td>42.0</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth">模型</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50_log.txt">日志</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>500</td>
      <td>0.083</td>
      <td>43.3</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth">模型</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r50-dc5_log.txt">日志</a></td>
      <td>159Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>500</td>
      <td>0.050</td>
      <td>43.5</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth">模型</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101_log.txt">日志</a></td>
      <td>232Mb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DETR-DC5</td>
      <td>R101</td>
      <td>500</td>
      <td>0.097</td>
      <td>44.9</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth">模型</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detr/logs/detr-r101-dc5_log.txt">日志</a></td>
      <td>232Mb</td>
    </tr>
  </tbody>
</table>

COCO val5k 评估结果可以在[这里](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918)找到。

这些模型也可以通过 torch hub 获取，要加载预训练的 DETR R50 模型，只需执行：
```python
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
```

COCO 全景分割 val5k 模型：
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名称</th>
      <th>骨干网络</th>
      <th>框 AP</th>
      <th>分割 AP</th>
      <th>PQ</th>
      <th>下载链接</th>
      <th>大小</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DETR</td>
      <td>R50</td>
      <td>38.8</td>
      <td>31.1</td>
      <td>43.4</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth">下载</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DETR-DC5</td>
      <td>R50</td>
      <td>40.2</td>
      <td>31.9</td>
      <td>44.6</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-panoptic-da08f1b1.pth">下载</a></td>
      <td>165Mb</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DETR</td>
      <td>R101</td>
      <td>40.1</td>
      <td>33</td>
      <td>45.1</td>
      <td><a href="https://dl.fbaipublicfiles.com/detr/detr-r101-panoptic-40021d53.pth">下载</a></td>
      <td>237Mb</td>
    </tr>
  </tbody>
</table>

查看我们的[全景分割 Colab](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)了解如何使用和可视化 DETR 的全景分割预测。

# Notebooks

我们在 Colab 中提供了一些 notebook 来帮助你理解 DETR：
* [DETR 实践 Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb)：展示如何从 hub 加载模型，生成预测，然后可视化模型的注意力（类似于论文中的图）
* [独立 Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)：在这个 notebook 中，我们演示了如何用 50 行 Python 代码从头实现一个简化版的 DETR，然后可视化预测结果。如果你想更好地理解架构并在深入研究代码库之前进行探索，这是一个很好的起点。
* [全景分割 Colab Notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)：演示如何使用 DETR 进行全景分割并绘制预测结果。

# 使用方法 - 目标检测
DETR 中没有额外的编译组件，包依赖也很少，所以代码使用起来非常简单。我们提供了通过 conda 安装依赖的说明。
首先，在本地克隆仓库：
```
git clone https://github.com/facebookresearch/detr.git
```
然后，安装 PyTorch 1.5+ 和 torchvision 0.6+：
```
conda install -c pytorch pytorch torchvision
```
安装 pycocotools（用于 COCO 评估）和 scipy（用于训练）：
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
这样就完成了，应该可以开始训练和评估检测模型了。

（可选）要处理全景分割，请安装 panopticapi：
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## 数据准备

从 [http://cocodataset.org](http://cocodataset.org/#download) 下载并解压 COCO 2017 训练和验证图像及标注。
我们期望的目录结构如下：
```
path/to/coco/
  annotations/  # 标注 json 文件
  train2017/    # 训练图像
  val2017/      # 验证图像
```

## 训练
要在单个节点上使用 8 个 GPU 训练基线 DETR 300 轮，运行：
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco 
```
一个轮次需要 28 分钟，所以在单个有 8 个 V100 显卡的机器上训练 300 轮大约需要 6 天。
为了便于复现我们的结果，我们提供了[结果和训练日志](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)，
用于 150 轮训练（在单个机器上 3 天），达到 39.5/60.3 AP/AP50。

我们使用 AdamW 训练 DETR，在 transformer 中设置学习率为 1e-4，在骨干网络中为 1e-5。
使用水平翻转、缩放和裁剪进行数据增强。
图像被缩放到最小尺寸 800 和最大尺寸 1333。
transformer 使用 0.1 的 dropout 进行训练，整个模型使用 0.1 的梯度裁剪进行训练。

## 评估
要在单个 GPU 上评估 DETR R50 在 COCO val5k 上的性能，运行：
```
python main.py --batch_size 2 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /path/to/coco
```
我们在这个[gist](https://gist.github.com/szagoruyko/9c9ebb8455610958f7deaa27845d7918)中提供了所有 DETR 检测模型的结果。
请注意，数值会根据每个 GPU 的批量大小（图像数量）而变化。
非 DC5 模型使用批量大小 2 训练，DC5 使用 1，
所以如果使用超过 1 张图像/GPU 评估，DC5 模型的 AP 会显著下降。

## 多节点训练
通过 Slurm 和 [submitit](https://github.com/facebookincubator/submitit) 可以进行分布式训练：
```
pip install submitit
```
在 4 个节点上训练基线 DETR-6-6 模型 300 轮：
```
python run_with_submitit.py --timeout 3000 --coco_path /path/to/coco
```

# 使用方法 - 分割

我们展示了将 DETR 扩展到预测分割掩码相对简单。我们主要展示了强大的全景分割结果。

## 数据准备

对于全景分割，除了 coco 数据集（见上面的 coco 数据集说明）外，你还需要全景标注。你需要下载并解压[标注](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)。
我们期望的目录结构如下：
```
path/to/coco_panoptic/
  annotations/  # 标注 json 文件
  panoptic_train2017/    # 训练全景标注
  panoptic_val2017/      # 验证全景标注
```

## 训练

我们建议分两个阶段训练分割：首先训练 DETR 检测所有框，然后训练分割头。
对于全景分割，DETR 必须学习检测 stuff 和 things 类别的框。你可以在单个节点上使用 8 个 GPU 训练 300 轮：
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic --dataset_file coco_panoptic --output_dir /output/path/box_model
```
对于实例分割，你可以简单地训练一个普通的框模型（或使用我们提供的预训练模型）。

一旦你有了框模型检查点，你需要冻结它，然后单独训练分割头。
对于全景分割，你可以在单个节点上使用 8 个 GPU 训练 25 轮：
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --masks --epochs 25 --lr_drop 15 --coco_path /path/to/coco  --coco_panoptic_path /path/to/coco_panoptic  --dataset_file coco_panoptic --frozen_weights /output/path/box_model/checkpoint.pth --output_dir /output/path/segm_model
```
对于仅实例分割，只需从上述命令行中移除 `dataset_file` 和 `coco_panoptic_path` 参数。

# 许可证
DETR 在 Apache 2.0 许可证下发布。更多信息请参见 [LICENSE](LICENSE) 文件。

# 贡献
我们热烈欢迎你的拉取请求！更多信息请参见 [CONTRIBUTING.md](.github/CONTRIBUTING.md) 和 [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)。 