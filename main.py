import collections
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
import mxnet
import time
import d2lzh as d2l
from mxnet.contrib import text
import gluonnlp
import re
from tqdm import tqdm


# 构建BertClass类
class BertClass(nn.Block):
    def __init__(self, bert,max_seq_len,out_put_num, ctx=None, **kwargs):
        super(BertClass, self).__init__(**kwargs)
        self.ctx = ctx
        self.max_seq_len = max_seq_len
        self.bert = bert
        self.output_dense = nn.Dense(out_put_num*max_seq_len)

    def forward(self, content, token_types, valid_len):
        bert_output = self.bert(content, token_types, valid_len)
        bert_output_shape = bert_output.shape
        bert_output = bert_output.reshape(bert_output_shape[0],bert_output_shape[1]*bert_output_shape[2])
        output = self.output_dense(bert_output)
        output = output.reshape(bert_output_shape[0],bert_output_shape[1],-1)
        return output


# 构建BiRNN类
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.T)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = nd.concat(outputs[0], outputs[-1])
        outs = self.decoder(encoding)
        return outs


# 本函数已保存在d2lzh包中方便以后使用
def download_imdb(data_dir='../data'):
    # url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    # sha1 = '01ada507287d82875905620988597833ad4e0903'
    # fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open("../data/aclImdb_v1.tar.gz", 'r') as f:
        f.extractall(data_dir)


# 读取训练数据集
def read_BIO(file_path):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    line_list = []
    line_label_list = []
    f = open(file_path, 'r', errors='ignore', encoding='utf-8')
    for line in tqdm(f):
        if line[0] != "\n":
            line_list.append(line[0])
            word_label = line[2:]
            line_label_list.append(word_label.replace("\n","").replace(" ",""))
        else:
            data.append([line_list,line_label_list])
            line_list = []
            line_label_list = []
    if len(line_list) != 0 and len(line_list) ==len(line_label_list):
        data.append([line_list, line_label_list])
    return data


# 读取测试数据集
def read_BIO_test(file_path):
    test_id_list = []
    data = []
    id_list = []
    line_list = []
    f = open(file_path, 'r', errors='ignore', encoding='utf-8')
    for line in tqdm(f):
        if line[0] != "\n":
            line_list.append(line[9])
            id = line[0:8]
        else:
            data.append(line_list)
            test_id_list.append(id)
            line_list = []
            id_list = []
    if len(line_list) == 1:
        print("ss:",line_list)
    if len(line_list) != 0 :
        data.append(line_list)
    counter = collections.Counter(test_id_list)
    id_vocab =  text.vocab.Vocabulary(counter)
    ids = id_vocab.to_indices(test_id_list)
    return zip(data,ids),id_vocab


def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]


def get_vocab_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)


def get_vocab_csv(data):
    list_date = [list(word) for _, word in data]
    counter = collections.Counter([word for line in list_date for word in line])
    return text.vocab.Vocabulary(counter, min_freq=5)


def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh包中方便以后使用
    max_l = 250 # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels


def preprocess_imdb_csv(data, ch_vocab,max_len):  # 本函数已保存在d2lzh包中方便以后使用
    '''
    将data数据去掉特殊符号，转换成打完pad的ndarray，简而言之就是preprocess_imdb函数加上去掉特殊符号并将vocab改成了ch_vocab，本注释中的list特指一维的ndarray
    :param data:句子list
    :param ch_vocab:bert返回的字典，可以把一个字转换成一个id
    :return: 注意下面返回数据加了s，比如contents要返回content的list，参照preprocess_imdb中的返回值
        contents: content单行数据示例： 你好世界-->[23,32,44,55,0,0,0...]
        token_types:token_type,bert传入参数的mask，暂时定义为 一个list:0*maxlen
        valid_lens：valid_len，句子的实际长度，当句子大于maxlen时，长度为maxlen，小于maxlen时是pad前的实际长度
    '''
    max_len -= 2 # 留出起始，结束位标记位
    content_list, token_type_list, valid_len_list,label_list = [],[],[],[]
    def pad(source):
        content = re.sub("[ \n\t\\n\u3000]", " ", source)
        content = re.sub("[?？]+", "？", content)
        content = [char for char in content]
        content = ch_vocab(content)
        if  len(content) > max_len:
            content = content[:max_len]
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            valid_len = max_len + 2
        else:
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            valid_len = len(content)
            content = content + [ch_vocab[ch_vocab.padding_token]] * (max_len + 2 - valid_len )
        content_list.append(content)
        token_type_list.append([0]*valid_len + [1]*(max_len + 2 - valid_len))
        valid_len_list.append(valid_len)
        return
    for content,label in data:
        pad(content)
        label_list.append(label)
    for i in range(len(content_list)):
        if len(content_list[i]) != 250:
            print(i,len(content_list[i]))
    contents = nd.array(content_list)
    token_types = nd.array(token_type_list)
    valid_lens = nd.array(valid_len_list)
    labels = nd.array([score for _, score in data])
    return contents, token_types, labels,valid_lens


def preprocess_imdb_BIO(data, ch_vocab,label_vocab,max_len):  # 本函数已保存在d2lzh包中方便以后使用
    '''
    将data数据去掉特殊符号，转换成打完pad的ndarray，简而言之就是preprocess_imdb函数加上去掉特殊符号并将vocab改成了ch_vocab，本注释中的list特指一维的ndarray
    :param data:句子list
    :param ch_vocab:bert返回的字典，可以把一个字转换成一个id
    :return: 注意下面返回数据加了s，比如contents要返回content的list，参照preprocess_imdb中的返回值
        contents: content单行数据示例： 你好世界-->[23,32,44,55,0,0,0...]
        token_types:token_type,bert传入参数的mask，暂时定义为 一个list:0*maxlen
        valid_lens：valid_len，句子的实际长度，当句子大于maxlen时，长度为maxlen，小于maxlen时是pad前的实际长度
    '''
    max_len -= 2 # 留出起始，结束位标记位
    content_list, token_type_list, valid_len_list,label_list = [],[],[],[]
    def pad(source,labels):
        content = ch_vocab(source)
        content_label =  [label_vocab.to_indices(x) for x in labels]
        if  len(content) > max_len:
            content = content[:max_len]
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            content_label =[0] + content_label[:max_len] + [0]
            valid_len = max_len + 2
        else:
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            valid_len = len(content)
            content = content + [ch_vocab[ch_vocab.padding_token]] * (max_len + 2 - valid_len )
            content_label = [0] + content_label + [0] + [0] *(max_len + 2 - valid_len )
        content_list.append(content)
        token_type_list.append([0]*valid_len + [1]*(max_len + 2 - valid_len))
        valid_len_list.append(valid_len)
        label_list.append(content_label)
        return
    for content,label in data:
        pad(content,label)
    contents = nd.array(content_list)
    token_types = nd.array(token_type_list)
    valid_lens = nd.array(valid_len_list)
    labels = nd.array(label_list)
    return contents, token_types, labels,valid_lens


def preprocess_imdb_BIO_test(data, ch_vocab,max_len):  # 本函数已保存在d2lzh包中方便以后使用
    max_len -= 2 # 留出起始，结束位标记位
    content_list, token_type_list, valid_len_list,label_list = [],[],[],[]
    id_list = []
    def pad(source,id):
        content = ch_vocab(source)
        if  len(content) > max_len:
            content = content[:max_len]
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            valid_len = max_len + 2
        else:
            content = [ch_vocab(ch_vocab.cls_token)] + content + [ch_vocab(ch_vocab.sep_token)]
            valid_len = len(content)
            content = content + [ch_vocab[ch_vocab.padding_token]] * (max_len + 2 - valid_len )
        content_list.append(content)
        token_type_list.append([0]*valid_len + [1]*(max_len + 2 - valid_len))
        valid_len_list.append(valid_len)
        id_list.append(id)
        return
    i = 0
    for content,id in data:
        pad(content,id)
    contents = nd.array(content_list)
    token_types = nd.array(token_type_list)
    valid_lens = nd.array(valid_len_list)
    ids = nd.array(id_list)
    return contents, token_types,valid_lens,ids


def predict_sentiment(net, vocab, sentence,ctx = None):
    sentence = nd.array(vocab.to_indices(sentence), ctx=ctx)
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'


def get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


# 评估模型精度
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    ctx = net.ctx
    for Xs, masks, ys ,valid_lens in data_iter:

        Xs = Xs.as_in_context(ctx)
        masks = masks.as_in_context(ctx)
        valid_lens = valid_lens.as_in_context(ctx)
        ys = ys.as_in_context(ctx)
        batch  = len(ys)
        y_hats = net(Xs, masks, valid_lens)
        ys = ys.reshape(-1, )
        ys_shape = ys.shape
        y_hats = y_hats.reshape(ys_shape[0], -1)
        this_test_acc = (y_hats.argmax(axis=1) == ys)
        acc_sum += this_test_acc.sum().asscalar()
        n += batch
    return acc_sum / n


# 测试
def do_test(data_iter,ch_vocab,id_vocab,label_vocab, net):
    acc_sum, n = 0.0, 0
    ctx = net.ctx
    ss = 0
    out_file = open("result.txt", "w")
    start_time = time.time()
    for Xs, masks ,valid_lens,ids in data_iter:
        Xs = Xs.as_in_context(ctx)
        masks = masks.as_in_context(ctx)
        valid_lens = valid_lens.as_in_context(ctx)
        y_hats = net(Xs, masks, valid_lens)
        y_hats_shape = y_hats.shape
        batch = y_hats_shape[0]
        y_hats = y_hats.reshape(y_hats_shape[0]  *y_hats_shape[1], -1)
        this_test_acc  = y_hats.argmax(axis=1)
        this_test_acc = this_test_acc.reshape(y_hats_shape[0]  , y_hats_shape[1])
        this_test_acc = this_test_acc.asnumpy().tolist()
        valid_lens = valid_lens.asnumpy().tolist()
        ids_list = ids.asnumpy().tolist()
        Xs = Xs.asnumpy().tolist()
        for i in range(y_hats_shape[0]):
            for j in range(int(valid_lens[i] -1)):
                if j == 0:
                    continue
                word = ch_vocab.idx_to_token[int(Xs[i][j])]
                word_type = label_vocab.idx_to_token[int(this_test_acc[i][j])]
                id = id_vocab.idx_to_token[int(ids_list[i])]
                out_file.write("%s %s %s\n"%(id,word,word_type))
            out_file.write("\n")
        ss += 1
        if ss % 1000 == 0:
            print("num:%d ,time %.1f sec",(ss,time.time() - start_time))

    out_file.close()
    return


# 训练
def train(train_iter, test_iter, net, loss, trainer,bert_trainer, ctx, num_epochs):
    """Train and evaluate a model."""
    print('training on', ctx)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        ii = 0
        #contents, token_types, labels,valid_lens
        for Xs, masks, ys ,valid_lens in train_iter:
            Xs = Xs.as_in_context(ctx)
            masks = masks.as_in_context(ctx)
            valid_lens = valid_lens.as_in_context(ctx)
            ys = ys.as_in_context(ctx)
            ls = []
            with mxnet.autograd.record():
                y_hats = net(Xs,masks,valid_lens)
                ys = ys.reshape(-1,)
                ys_shape = ys.shape
                y_hats = y_hats.reshape(ys_shape[0],-1)
                ls = loss(y_hats,ys)
                ls_ave = nd.sum(ls) / len(ls)
            batch = len(ls)
            ls_ave.backward()
            loss_scalar = ls_ave.asscalar()
            trainer.step(1)
            bert_trainer.step(1)
            train_l_sum += loss_scalar * batch
            this_train_acc = (y_hats.argmax(axis=1) == ys)
            train_acc_sum += this_train_acc.sum().asscalar()
            m += batch
            ii += 1
            if (ii % 100) == 0:
                print("epoch %d,num %d, loss %.4f, train acc %.3f, "
                      "time %.1f sec"
                      % (epoch + 1, ii,train_l_sum/m, train_acc_sum / m,
                         time.time() - start))
            if ii % 2000 == 0:
                print("dev acc %.3f "%evaluate_accuracy(test_iter,net))
        print("epoch end %d, loss %.4f, train acc %.3f, dev acc %.3f,"
              "time %.1f sec\n"
              % (epoch + 1, train_l_sum/m, train_acc_sum / m,evaluate_accuracy(test_iter,net),
                 time.time() - start))
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
        #       'time %.1f sec'
        #       % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
        #          time.time() - start))
        # d2l.train()


if __name__ == "__main__":
    useModel = True #加载已经训练好的模型
    run_dev_test = True #运行模型，跑一下当前模型验证集的正确率，不训练(useModel为true时生效)
    num_epochs = 2 #训练轮数
    save_model = False  #是否保存模型
    train_model = False # 是否训练模型
    run_test = False
    use_GPU = False #是否使用GPU
    filename = './model/mlp.params'
    # download_imdb()
    if use_GPU:
        ctx = d2l.try_gpu()
    else:
        ctx = mxnet.cpu()
    # train_data, dev_data = read_BIO('data/TrainData_BIO3.txt'), read_BIO('data/TrainData_BIO3.txt')
    all_data = read_BIO('data/newTrainData_BIO.txt')
    all_test_data,id_vocab = read_BIO_test('data/newTest_Data_BIO.txt')
    train_end_index = int(len(all_data) * 0.97)
    train_data = all_data[:train_end_index]
    dev_data = all_data[train_end_index+1:]
    label_vocab = get_vocab_csv(train_data)
    batch_size = 32
    max_len = 100
    ebmed_size, num_hiddens, num_layers = 100, 100, 2
    # 载入预训练语言模型并获取其中词向量
    ch_bert, ch_vocab = gluonnlp.model.get_model("bert_12_768_12",
                                                 dataset_name="wiki_cn_cased",
                                                 pretrained=True,
                                                 ctx=ctx,
                                                 use_pooler=False,
                                                 use_decoder=False,
                                                 use_classifier=False)
    if train_model:
        train_set = gdata.ArrayDataset(*preprocess_imdb_BIO(train_data, ch_vocab,label_vocab,max_len))
        train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
        print('#train batches:', len(train_iter))
    if run_test:
        test_set = gdata.ArrayDataset(*preprocess_imdb_BIO_test(all_test_data, ch_vocab, max_len))
        test_iter = gdata.DataLoader(test_set, batch_size)
        print('#test_iter batches:', len(test_iter))
    dev_set = gdata.ArrayDataset(*preprocess_imdb_BIO(dev_data, ch_vocab, label_vocab, max_len))
    dev_iter = gdata.DataLoader(dev_set, batch_size)

    net = BertClass(bert=ch_bert,
                      max_seq_len=max_len, out_put_num= len(label_vocab) , ctx=ctx)
    if(useModel == True):
        net.load_parameters(filename,ctx=ctx)
        if(run_dev_test == True):
            print('dev acc %.3f\n'%evaluate_accuracy(dev_iter, net))
        if run_test == True:
            do_test(test_iter,ch_vocab,id_vocab,label_vocab,net)
    else:
        net.output_dense.initialize(init.Xavier(), ctx=ctx) #初始化模型
    if train_model:
        net.hybridize()
        # build trainer
        bert_trainer = gluon.Trainer(ch_bert.collect_params(),
                                         'adam', {"learning_rate": 1e-5})
        trainer = gluon.Trainer(net.collect_params("dense*"), 'adam',
                                {"learning_rate": 3e-5})
        loss = gloss.SoftmaxCrossEntropyLoss()
        train(train_iter, dev_iter, net, loss, trainer,bert_trainer, ctx, num_epochs)
    if train_model and save_model:
        net.save_parameters(filename)
