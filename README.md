# CHAN-DST (수정)
여러 error들을 디버깅

[[원본깃헙링크](https://github.com/smartyfh/CHAN-DST)]

Code for our ACL 2020 paper: **A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking. Yong Shan, Zekang Li, Jinchao Zhang, Fandong Meng, Yang Feng, Cheng Niu, Jie Zhou. ACL 2020 *(Long)***. [[arxiv](https://arxiv.org/abs/2006.01554)]
[[논문링크](https://www.aclweb.org/anthology/2020.acl-main.563.pdf)]

## Abstract
Recent studies in dialogue state tracking (DST) leverage historical information to determine states which are generally represented as slot-value pairs. However, most of them have limitations to efficiently exploit relevant context due to the lack of a powerful mechanism for modeling interactions between the slot and the dialogue history. Besides, existing methods usually ignore the slot imbalance problem and treat all slots indiscriminately, which limits the learning of hard slots and eventually hurts overall performance. In this paper, we propose to enhance the DST through employing a contextual hierarchical attention network to not only discern relevant information at both word level and turn level but also learn contextual representations. We further propose an adaptive objective to alleviate the slot imbalance problem by dynamically adjust weights of different slots during training. Experimental results show that our approach reaches 52.68% and 58.55% joint accuracy on MultiWOZ 2.0 and MultiWOZ 2.1 datasets respectively and achieves new state-of-the-art performance with considerable improvements (+1.24% and +5.98%).

<p align="center"><img src="https://i.loli.net/2020/06/05/rsEHlLake37SdoY.jpg" width="80%" class="center"/></p>

## Requirements
* python 3.6
* pytorch >= 1.0
* Install python packages:
  - ``pip install -r requirements.txt``


## Usages
### Data Preprocessing
We conduct experiments on the following datasets:
* MultiWOZ 2.0 [Download](https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y)
* MultiWOZ 2.1 [Download](https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y)

We use the same preprocessing steps for both datasets. For example, preprocessing Multiwoz 2.0:
```bash
$ pwd
/home/user/chan-dst
# download multiwoz 2.0 dataset
$ wget https://www.repository.cam.ac.uk/bitstream/handle/1810/280608/MULTIWOZ2.zip?sequence=3&isAllowed=y -O multiwoz2.0.zip
# preprocess datasets for training DST and STP jointly
# $ unzip -j multiwoz2.0.zip -d data/multiwoz-update/original
$ unzip -j MULTIWOZ2.zip -d data/multiwoz-update/original
$ cd data/multiwoz-update/original
$ mv ontology.json ..
$ python convert_to_glue_format.py
# preprocessing datasets for fine-tuning DST with adaptive objective
# $ unzip -j multiwoz2.0.zip -d data/multiwoz/original
$ unzip -j MULTIWOZ2.zip -d data/multiwoz/original
$ cd data/multiwoz/original
$ mv ontology.json ..
$ python convert_to_glue_format.py
```

For Multiwoz 2.1, replace the corresponding directories with `multiwoz2.1-update` and `multiwoz2.1`, respectively.

### Train
Take MultiWOZ 2.0 as an example.

1. Pre-training

    ```bash
    bash run_multiwoz2.0.sh
    ```
    
    ```bash
    bash run_MultiWOS.sh
    ```

2. Fine-tuning

    ```bash
    bash run_multiwoz2.0_finetune.sh
    ```

    ```bash
    bash run_MultiWOS_finetune.sh
    ```

## Citation
If you find this code useful, please cite as:

```
@inproceedings{shan2020contextual,
  title={A Contextual Hierarchical Attention Network with Adaptive Objective for Dialogue State Tracking},
  author={Shan Yong, Li Zekang, Zhang Jinchao, Meng Fandong, Feng Yang, Niu Cheng, Zhou Jie},
  booktitle={Proceedings of the 58th Conference of the Association for Computational Linguistics},
  year={2020}
}
```
---
## 수정사항
CHAN-DST 공식 레포 코드를 가져와서 실제 훈련시 사용했던 MultiWOZ 데이터에 대해서 그대로 돌렸을 때 많은 에러들이 발생, (pytorch 버전이 맞지 않는 이슈, nonzero메소드 변경, [mask 관련 turn_mask 등]에서 tensor byte→bool type변경 등)

CHAN-DST 코드를 MultiWOS 데이터(우리데이터)에 적용하기 위해 데이터의 형식을 바꾸어주었음 (tsv 파일로 변경), 내부 코드 또한 huggingface tranformers의 pretrained BertModel을 가져오기 위해 변경해주었음

CHAN-DST 코드를 MultiWOS 데이터에 적용해보았는데, model_adaptive_finetune.py 만 적용해볼 수 있었다. (CHAN-DST는 pretrained → fine-tuning까지 모두 거치는 과정이 포함되어있는데 불가능하였음)

CHAN-DST (only fine-tuning) 결과 도출 후 inference, LB 제출 & MultiWOZ와 달리 MultiWOS(우리데이터)에서의 한계점을 파악하였음