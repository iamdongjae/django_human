import os
import torch
import numpy as np
import gluonnlp as nlp
from transformers import AdamW

from kobert.bert_module.bert_dataset import BERTDataset
from kobert.bert_module.bert_classifier import BERTClassifier
from kobert_tokenizer import KoBERTTokenizer


# 모델있는 경로로 변경
#os.chdir('./models1/')
# GPU가 있으면 cuda, 없으면 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = 'kobert/fine_model/'
# 코랩 GPU로 생성된 모델을 CPU 모델로 변환해서 읽기 위해 map_locaion=device 매개변수를 추가한다.
model = torch.load(path+'7emotions_model_2.pt', map_location=device)  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(path+'7emotions_model_state_dict_2.pt', map_location=device))  # state_dict를 불러 온 후, 모델에 저장
checkpoint = torch.load(path+'7emotions_all_2.tar', map_location=device)   # dict 불러오기
model.load_state_dict(checkpoint['model'])

# 파라미터 설정
max_len = 64
batch_size = 64
learning_rate =  5e-5

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tok = tokenizer.tokenize
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# optimizer 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)
optimizer.load_state_dict(checkpoint['optimizer'])   

def predict(predict_sentence): # input = 감정분류하고자 하는 sentence

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False) # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out: # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")

        #print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")
        return test_eval[0]

#질문 무한반복하기! 0 입력시 종료
#end = 1
#while end == 1 :
#    sentence = input("하고싶은 말을 입력해주세요 : ")
#    if sentence == '0' :
#        break
#    predict(sentence)
#    print("\n")