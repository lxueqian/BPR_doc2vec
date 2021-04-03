#coding:utf-8
#使用doc2vec 判断文档相似性
from gensim import models,corpora,similarities
import jieba.posseg as pseg
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
import os
import time
import json
import numpy as np

stop = [line.strip() for line in open('stopword.txt').readlines()]

# print('stop:',stop)

def generate_file():
	with open('/home/lixueqian/lxq/review_per_user/live_stats.json','r') as f:
		live_dic = json.loads(f.read())
	
	live_id_dic = {}
	print(len(live_dic))

	i = 0	
	for _id,live in live_dic.items():
		print(_id,'begin.....')
		temp = live['tags'][0]+' '+live['subject']+'\n'+live['description']
		with open('./file/'+str(i)+'.txt','w') as f:
			f.write(json.dumps(temp,ensure_ascii=False))
		live_id_dic[i] = _id
		i += 1
	
	print('live文件总个数：',i)
	with open('live_id_map.txt','w') as f:
		f.write(json.dumps(live_id_dic))

def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret

def read_file():
	#读取文件
	raw_documents=[]
	walk = os.walk(os.path.realpath("./file"))
	for root, dirs, files in walk:
		files.sort(key=lambda x: int(x[:-4]))
		# print(files)
		for name in files:
			f = open(os.path.join(root, name), 'r')
			# raw = str(os.path.join(root, name))+" "
			raw = " "
			raw += f.read()
			raw_documents.append(raw)
	# print(raw_documents)
	return raw_documents

def construct_vocabulary(raw_documents):
	#构建语料库
	corpora_documents = []
	doc=[]            #输出时使用，用来存储未经过TaggedDocument处理的数据，如果输出document，前面会有u
	for i, item_text in enumerate(raw_documents):
		# print(item_text)
		words_list=[]
		item=(pseg.cut(item_text))

		for j in list(item):
			words_list.append(j.word)
		words_list=a_sub_b(words_list,list(stop))

		# print(words_list)
		# time.sleep(10)
		document = TaggedDocument(words=words_list, tags=[i])
		# print(document)
		# time.sleep(10)
		corpora_documents.append(document)
		doc.append(words_list)

	with open('doc.txt','w') as f:
		f.write(json.dumps(doc,ensure_ascii=False))

	return corpora_documents

def model(corpora_documents):
	# 创建model
	model = Doc2Vec(vector_size=70, window=4, min_count=2, epochs=20)
	model.build_vocab(corpora_documents)
	model.train(corpora_documents,total_examples=6616, epochs=20)
	print('#########', model.vector_size)
	# sims = model.docvecs.most_similar([inferred_vector], topn=3)

	model.save('doc2vec.bin')

def predict():
	model = Doc2Vec.load('doc2vec.bin')
	with open('doc.txt','r') as f:
		doc = json.loads(f.read())

	
	# embedding_vector = []
	# for raw in doc:		
	# 	embedding_vector.append(model.infer_vector(raw))

	# embedding_vector = np.array(embedding_vector)
	# np.save('embedding_vector.npy',embedding_vector)
	
	print(doc[19])
	print(doc[33])
	inferred_vector1 = model.infer_vector(doc[19])
	inferred_vector2 = model.infer_vector(doc[33])
	inferred_vector = 4*inferred_vector1 + inferred_vector2
	sims = model.docvecs.most_similar([inferred_vector], topn=10)
	print(sims)  #sims是一个tuples,(index_of_document, similarity)
	for i in sims:
		similar=""
		print('################################')
		print(i[0])
		for j in doc[i[0]]:
			similar+=j
		print(similar)
		time.sleep(5)
	# break

if __name__ == '__main__':
	# generate_file()
	# raw_documents = read_file()
	# corpora_documents = construct_vocabulary(raw_documents)
	# model(corpora_documents)
	predict()

