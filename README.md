对目标检测论文按照解决的问题进行分类，参考:    
https://github.com/hoya012/deep_learning_object_detection      
https://github.com/dutxubo/ObjectDetectionImbalance



# 1. Class Imbalance

​	类别不平衡问题在计算机视觉领域是一个普遍存在的问题，具体的可以分为前景-背景不平衡和前景-前景不平衡。

## 1.1.  Foreground-Backgorund Class Imbalance
    目标检测中背景相对于前景数量众多，尤其是在anchor-based和anchor-free检测器中，将注意力集中在有效的正负样本上能够有效的提升训练的性能。
    可以通过Hard Sampling的方式，利用某种方式硬性的采样，如OHEM、
    也可以通过Soft Sampling对正负样本进行加权，如Focal Loss根据损失加权，GHM根据梯度进行加权
- Hard Sampling Methods   
	- Random Sampling
	- Hard Example Mining
		- Online Hard Example Mining, CVPR 2016, [[paper]](https://zpascal.net/cvpr2016/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
		-  IoU-based Sampling, CVPR 2019, [[paper\]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.pdf) 
- Soft Sampling Methods   
	- Focal Loss, ICCV 2017, [[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
	- Gradient Harmonizing Mechanism, AAAI 2019, [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/4877)
- Sampling-Free Methods
	- Are Sampling Heuristics Necessary in Object Detectors?, arXiv 2019, [[paper]](https://arxiv.org/pdf/1909.04868.pdf)
- Generative Methods 
	- Adversarial Faster-RCNN, CVPR 2017, [[paper]](http://zpascal.net/cvpr2017/Wang_A-Fast-RCNN_Hard_Positive_CVPR_2017_paper.pdf) 

## 1.2. Foreground-Foreground Class Imbalance
    前景-前景不平衡问题是一个长尾问题。
- Fine-tuning Long Tail Distribution for Obj.Det., CVPR 2016, [[paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Ouyang_Factors_in_Finetuning_CVPR_2016_paper.pdf)
- PSIS, arXiv 2019, [[paper\]](https://arxiv.org/pdf/1906.00358.pdf)
- OFB Sampling, WACV 2020, [[paper\]](https://arxiv.org/abs/1909.09777)

# 2. Scale Problem
## 2.1 FPN
    	FPN已经是处理尺度问题的一个标准组件了，不再详细阐述。不过FPN中隐含的一个问题是不同大小的目标应该映射到哪一个level特征上？在每个level上映射全部目标并不合理，通常的做法是将不同尺度的目标按照先验划分到一个level上，而论文FSAF提出对于同一个目标在多个level上进行预测性能会有所提示。
    	SAPD通过一个meta-selection net来得到目标锚点分配在各level的权重;ATSS利用IOU的统计信息（均值+标准差）在各level上通过IOU确定正负样本，避免了根据尺度先验来分配正负样本。
- FPN, CVPR 2017, [[paper]](https://zpascal.net/cvpr2017/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)
- SAPD: Soft Anchor-Point Object Detection, 2019, [[paper]]( https://arxiv.org/abs/1911.12448 )
- ATSS: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection, 2019, [[paper]]( https://arxiv.org/abs/1912.02424 ), [[code]]( https://github.com/sfzhang15/ATSS )

## 2.2 针对尺度特殊的训练
    	一般使用multi-scale training/testing的技巧提升结果。
    	SNIP提出在训练时对于不同尺度的输入图像，目标也应该在一定范围内，即忽略掉过大或者过小的proposal；在测试中，建立大小不同的Image Pyramid，在每张图上都运行这样一个detector，同样只保留那些大小在指定范围之内的输出结果，最终在一起NMS。对比各种不同baseline，在COCO数据集上有稳定的3个点提升。
    	TridentNet使用三个不同dilate的分支训练，权值共享，在测试时只使用一个分支进行输出
- SNIP, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Singh_An_Analysis_of_CVPR_2018_paper.pdf)
- SNIPER: Efficient Multi-Scale Training，2018

- Scale Aware Trident Network, ICCV 2019, [[paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.pdf)
## 2.3 NAS搜索网络结构
    直接使用NAS搜索能够适应尺度变化的Backbone。
- SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization. 2019，[[paper]]( https://arxiv.org/abs/1912.05027 )

# 3. 更精确的定位
    为了更精确的定位目标，有以下几种方式
    一是改进回归函数，可以基于范数的方式进行改进，也可以基于iou的方式进行改进
    二是使用级联的方式逐步精确定位
    三是对可能包含目标位置处进行更密集的采样,如guided_anchoring,避免手动设置锚点框
## 3.1. 改进回归函数
- Lp norm based
	- Smooth L1, ICCV 2015, [[paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
	- Balanced L1, CVPR 2019, [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Pang_Libra_R-CNN_Towards_Balanced_Learning_for_Object_Detection_CVPR_2019_paper.html)
- IoU based
	- IoU Loss, ACM IMM 2016, [[paper]](https://arxiv.org/pdf/1608.01471.pdf)
	- Bounded IoU Loss, CVPR 2018, [[paper]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0794.pdf)
## 3.2. 级联逐步精确定位
- Cascade R-CNN, CVPR 2018,  [[paper\]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf) 
- Hierarchical Shot Detector, ICCV 2019,  [[paper\]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Hierarchical_Shot_Detector_ICCV_2019_paper.pdf) 
## 3.3.  注意力密集采样
- Guided Anchoring:  Region Proposal by Guided Anchoring , CVPR 2019, [[paper]]( https://arxiv.org/abs/1901.03278 )
- FreeAnchor: Learning to Match Anchors for Visual Object Detection, NeurIPS 2019, [[paper]]( https://arxiv.org/abs/1909.02466 )

# 4. 更好的提取特征
    使用DCN、DCN-v2等方式能够显著提高目标检测性能，近年来，在特征提取方面又有新的进展。
    之前的DCN都是弱监督的方式，reppoints创造性的使用边界框对DCN的偏移量进行监督。

-  Non-local Networks
-  Reppoints: Point Set Representation for Object Detection. ICCV 2019, [[paper]]( https://arxiv.org/abs/1904.11490 ), [[code]]( https://github.com/open-mmlab/mmdetection ) 

# 3. Augmentation
    人工设计数据增强方式，如cutout、mixup、copy-past等
    利用AutoML实现自动数据增强

- Augmentation for small object detection. 2019, [[paper]]( https://arxiv.org/abs/1902.07296 )
- AutoAugment: Learning Augmentation Policies from Data. CVPR 2019, [[paper]]( https://arxiv.org/abs/1805.09501 )
- Fast AutoAugment. 2019, [[paper]]( https://arxiv.org/abs/1905.00397 )
- Data Augmentation Revisited:Rethinking the Distribution Gap between Clean and Augmented Data. 2019, [[paper]]( https://arxiv.org/abs/1909.09148 )
- Learning Data Augmentation Strategies for Object Detection. 2019, [[paper]]( https://arxiv.org/abs/1906.11172 )
- RandAugment: Practical automated data augmentation with a reduced search space. 2019, [[paper]]( https://arxiv.org/abs/1909.13719 )








