import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, CpmTokenizer
from utils import set_logger, save_metric
from evaluate import load
from metrics import Perplexity, Bleu
from nltk import ngrams
from rouge import Rouge
import os
import json

# list of list
def generate(input_ids):
    output_ids = []
    
    for input_id in input_ids:
        cur_input_tensor = torch.tensor([input_id], dtype=torch.long)
        cur_output_id = model.generate(cur_input_tensor, max_length=500, eos_token_id=eod_id, topk=50, topp=0.9, do_sample=True, num_return_sequences=1, use_cache=True, repetition_penalty=1.4)
        output_ids.append(cur_output_id[0])
        
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs
            

#input_texts 是 list of str
def get_ppl(input_texts, model, tokenizer):
    perplexity = Perplexity() # ('perplexity', module_type='metric')
    results = perplexity.compute(input_texts=input_texts, model=model, tokenizer=tokenizer)
    return results

# predictions 是 list of str, references 是 list of list
def get_bleu(predictions, references, tokenizer):
    bleu = Bleu()
    processed_references = [[ref] for ref in references]
    results = bleu.compute(predictions=predictions, references=processed_references, tokenizer=tokenizer)
    return results

# 两个都是 List of str
def get_rouge(predictions, references):
    #! predictions 里面的中文字符串是每个字之间要有一个 space
    #rouge = load('rouge')
    processed_predictions = [' '.join(list(pred)) for pred in predictions]
    processed_references = [' '.join(list(ref)) for ref in references]
    results = Rouge().get_scores(refs=processed_predictions, hyps=processed_references)[0]
    # predictions = ["hello there", "general kenobi"]
    # references = ["hello there", "general kenobi"]
    # results = rouge.compute(predictions=processed_predictions,
    #                         references=processed_references)
    return results

# 都是 list of str
def get_BERTScore(predictions, references, model_type):
    bertscore = load("bertscore")
    # predictions = ["hello world", "general kenobi"]
    # references = ["goodnight moon", "the sun is shining"]
    # model_type="distilbert-base-uncased"
    results = bertscore.compute(predictions=predictions, references=references, model_type=model_type)
    return results

# list of str
def get_distinct(predictions):
    processed_predictions = [' '.join(list(pred)) for pred in predictions]
    result = {}
    for i in range(1, 5):
        all_ngram, all_ngram_num = {}, 0.
        for k, pred in enumerate(processed_predictions):
            ngs = ["_".join(c) for c in ngrams(pred.strip().split(), i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = len(all_ngram) / float(all_ngram_num)
    return result


#! Testing
def evaluate(preds, refs):
    bert_score = get_BERTScore(preds, refs, 'bert-base-chinese')
    ppl = get_ppl(refs, model, tokenizer)
    bleu = get_bleu(preds, refs, tokenizer)
    rouge = get_rouge(preds, refs)
    distinct = get_distinct(preds)
    return [ppl, bert_score, bleu, rouge, distinct]
    

def build_dataset(data_path, task):
    inputs, origin_text = [], []
    sample = 3
    count = 0
    
    with open(data_path) as f:
        for line in f:
            cur_json = json.loads(line)
            story = cur_json['story']
            half_idx = len(story) // 2
            story_input = story[:half_idx]
            story_ids = tokenizer.encode(story_input, add_special_tokens=False)
            
            if task == 'post-training':
                input_ids = story_ids
                inputs.append(input_ids)
                origin_text.append(story)
                
            elif task == 'fine-tuning':
                moral_ids = tokenizer.encode(cur_json['moral'], add_special_tokens=False)
                input_ids = moral_ids + [sep_id] + story_ids
                inputs.append(input_ids)
                origin_text.append(cur_json['moral'] + '<sep>' + story)
            
            count += 1
            if count >= sample:
                break
    return inputs, origin_text

if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=50, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.9, type=float, required=False, help='最高积累概率')
    parser.add_argument('--context_len', default=200, type=int, required=False, help='作文生成中，每一步生成时，参考的上文的长度')
    # parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚参数')
    parser.add_argument('--log_path', default='log/evaluating.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--model_path', type=str, default='model/zuowen_epoch40', help='模型存放位置')
    parser.add_argument('--test_dataset_path', type=str, default='data/outgen/test.jsonl', help='test dataset 数据集')
    parser.add_argument('--metric_output_path', type=str, default='model/zuowen_epoch40', help='metric 输出地址')
    parser.add_argument('--task', default='post-training', type=str, required=False, help='post-training or fine-tuning')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    device = 'cuda' if args.cuda else 'cpu'
    # device = 'cpu'

    # 创建日志对象
    logger = set_logger(args.log_path)

    # 加载tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)
    logger.info('start evaluating...')
    
    # #! 加载 test dataset
    
    inputs, origin_text = build_dataset(args.test_dataset_path, args.task)
    # results = evaluate(model, dataset)
    # input_texts = ['从前，兔子和狼是朋友，它们常在一起打猎。但是每次分配时，狼总是将较大的那一份给自己。兔子觉得这实在是太过分了。', '我今天过得很高兴']
    # preds = ['我今天很开心']
    # refs = ['我很开心']
    preds = generate(inputs)
    print(preds[0])
    
    print()
    print(origin_text[0])
    results = evaluate(preds, origin_text)
    # get_BERTScore(preds, refs, './pretrained_models/bert-base-chinese')
        


    save_metric(results, os.path.join(args.metric_output_path, 'test_metrics.json'))




