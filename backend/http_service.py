from pydoc import render_doc
import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel, CpmTokenizer, TextGenerationPipeline
from utils import top_k_top_p_filtering, set_logger
from os.path import join
from flask import Flask, redirect, url_for, request, jsonify, render_template

app = Flask(__name__, template_folder='../frontend/dist', static_folder='../frontend/dist/static')
app.config["JSON_AS_ASCII"] = False  # 防止返回中文乱码

def generate(moral, context):
    
    if moral:
        moral_ids = tokenizer.encode(moral, add_special_tokens=False)
        
    if context:
        context_ids = tokenizer.encode(context, add_special_tokens=False)

    
    if moral and context:
        input_ids = moral_ids + [sep_id] + context_ids
    
    elif moral and not context:
        input_ids = moral_ids + [sep_id] 
    
    elif not moral and context:
        input_ids = context_ids + [sep_id]

    else:
        input_ids = [sep_id]
        
    #! 只看200个
    if len(input_ids) > args.context_len:
        input_ids = input_ids[-args.context_len:]

    cur_input = torch.tensor([input_ids], dtype=torch.long)
    valid_count = 0
    valid_list_return = []
    punc = [',', '。', '!', '?','？','！']

    while True:
        #! 可能一下子加了两个
        if valid_count >= 3:
            break
        output_ids1 = model.generate(cur_input, max_length=100, eos_token_id=sen_end, num_beams=3, do_sample=True, num_return_sequences=1, use_cache=True)
        output_ids2 = model.generate(cur_input, max_length=100, eos_token_id=sen_medium, topk=200, topp=0.9, do_sample=True, num_return_sequences=1, use_cache=True)
        outputs = tokenizer.batch_decode([output_ids1[0], output_ids2[0]], skip_special_tokens=False)
        for output in outputs:
            try:
                output = output.split('<sep>')[1]
            except IndexError:
                output = output.split('<sep>')[0]
            output = output.replace('\n', '')
            #! 从 context 往后截取
            true_output_idx = len(context)
            true_output = output[true_output_idx:]
            
            if true_output and true_output not in punc:
                valid_list_return.append(true_output)
                valid_count += 1

    return valid_list_return

@app.route('/', methods=['POST', 'GET'], strict_slashes=False)
def index():
    return render_template('index.html')
    
@app.route('/story', methods=['POST', 'GET'], strict_slashes=False)
def story():
    if request.method == 'POST':
        data = request.json
        moral = data['moral']
        context = data['context']
        cand_list = generate(moral, context)
        cand_list = list(set(cand_list))
    return jsonify({'cand_list':cand_list})


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=200, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.85, type=float, required=False, help='最高积累概率')
    parser.add_argument('--context_len', default=200, type=int, required=False, help='作文生成中，每一步生成时，参考的上文的长度')
    # parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚参数')
    parser.add_argument('--port', type=int, default=8085, help='服务绑定的端口号')
    parser.add_argument('--log_path', default='log/http_service.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--model_path', type=str, default='model/zuowen_epoch40', help='模型存放位置')
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
    unk_id = tokenizer.unk_token_id
    sen_end = tokenizer.convert_tokens_to_ids("。")
    sen_medium = tokenizer.convert_tokens_to_ids(",")

    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)

    app.run(debug=True, host="0.0.0.0", port=args.port)
