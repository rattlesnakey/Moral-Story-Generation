import argparse
from utils import set_logger
from transformers import CpmTokenizer
import os
# import pickle
import json
from tqdm import tqdm


def preprocess():
    """
    对故事数据集进行预处理
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/preprocess.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--data_path', default='data/outgen/train.jsonl', type=str, required=False, help='数据集存放位置')
    parser.add_argument('--save_path', default='data/train.json', type=str, required=False, help='对训练数据集进行tokenize之后的数据存放位置')
    parser.add_argument('--type', default='post_train', type=str, required=False, help='post-train or fine-tune')
    parser.add_argument('--win_size', default=200, type=int, required=False, help='滑动窗口的大小，相当于每条数据的最大长度')
    parser.add_argument('--step', default=200, type=int, required=False, help='滑动窗口的滑动步幅')
    args = parser.parse_args()

    # 初始化日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")   # 文档结束符
    sep_id = tokenizer.sep_token_id

    # 读取作文数据集目录下的所有文件
    
    train_list = []
    logger.info("start tokenizing data")
    with open(args.data_path, 'r') as f:
        for line in f:
            line = line.strip()
            temp_dict = json.loads(line)
            story = temp_dict['story']
            story_token_ids = tokenizer.encode(story, add_special_tokens=False)
            if args.type == 'fine-tune':
                moral = temp_dict['moral']
                moral_token_ids = tokenizer.encode(moral, add_special_tokens=False)
                token_ids = moral_token_ids + [sep_id] + story_token_ids + [eod_id]
            
            elif args.type == 'post-train':
                token_ids = story_token_ids + [eod_id]
            
            # train_list.append(token_ids)
            # 对于每条数据，使用滑动窗口对其进行截断
            win_size = args.win_size
            step = args.step
            start_index = 0
            end_index = win_size
            
            while True:
                    data = token_ids[start_index:end_index]
                    if not data:
                        break
                    train_list.append(data)
                    start_index += step
                    end_index += step

    
    with open(args.save_path, 'w+') as f:
        json.dump(train_list, f)
    
    logger.info('done')

            
        
    # lens = [len(s) for s in train_list]
    # for l in lens:
    #     if l > 300:
    #         print(l)
    # print(max(lens))
    # for file in tqdm(os.listdir(args.data_path)):
    #     file = os.path.join(args.data_path, file)
    #     with open(file, "r", encoding="utf8")as reader:
    #         lines = reader.readlines()
    #         title = lines[1][3:].strip()    # 取出标题
    #         lines = lines[7:]   # 取出正文内容
    #         article = ""
    #         for line in lines:
    #             if line.strip() != "":  # 去除换行
    #                 article += line
    #         title_ids = tokenizer.encode(title, add_special_tokens=False)
    #         article_ids = tokenizer.encode(article, add_special_tokens=False)
    #         token_ids = title_ids + [sep_id] + article_ids + [eod_id]
    #         # train_list.append(token_ids)

    #         # 对于每条数据，使用滑动窗口对其进行截断
    #         win_size = args.win_size
    #         step = args.step
    #         start_index = 0
    #         end_index = win_size
    #         data = token_ids[start_index:end_index]
    #         train_list.append(data)
    #         start_index += step
    #         end_index += step
    #         while end_index+50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
    #             data = token_ids[start_index:end_index]
    #             train_list.append(data)
    #             start_index += step
    #             end_index += step

    # 序列化训练数据
    # with open(args.save_path, "wb") as f:
    #     pickle.dump(train_list, f)


if __name__ == '__main__':
    preprocess()


