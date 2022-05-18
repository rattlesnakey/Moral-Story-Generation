# Moral Story Generation
> 以道理为基础的故事生成系统，模型采用CPM，先基于故事Domain 数据集进行 Post-training，然后再基于 Moral Story 数据集:Moral[SEP]Story进行fine-tuning

## Data
* [Story Domain](https://cloud.tsinghua.edu.cn/d/0cf033b0c7c049be855d/?p=%2Foutgen&mode=list)
* [Moral Story](https://github.com/thu-coai/MoralStory)

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
```shell
cd backend
bash http_service.sh
```

