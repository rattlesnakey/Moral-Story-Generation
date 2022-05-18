# Moral Story Generation
> 以道理为基础的故事生成系统，模型采用CPM，先基于故事Domain 数据集进行 Post-training，然后再基于 Moral Story 数据集:Moral[SEP]Story进行fine-tuning

## Data
* [Story Domain](https://cloud.tsinghua.edu.cn/d/0cf033b0c7c049be855d/?p=%2Foutgen&mode=list)
* [Moral Story](https://github.com/thu-coai/MoralStory)
* 也可以到 node02  /home/work/zhanghengyuan/projects/story/CPM/data 文件夹里获取

## Setup
> 前端 Vue，后端 Flask
### Backend
```shell
cd backend
conda env create -f environment.yaml
```

### Frontend
```shell
cd frontend
npm install .

```
## Get Start
### Train
```shell
cd backend
bash train.sh
```

### run plantform
> 如果没有训练模型，可以先到 /home/work/zhanghengyuan/projects/story/CPM/model 文件夹下面获取模型文件
```shell
cd backend
bash http_service.sh
```

