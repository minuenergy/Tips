import gc
import unittest
from transformers import (
    AutoProcessor,
    LlavaConfig,
    LlavaForConditionalGeneration,
    is_torch_available,
    is_vision_available,
    AutoTokenizer
)
from transformers.testing_utils import require_bitsandbytes, require_torch, slow, torch_device
from transformers import BitsAndBytesConfig
from PIL import Image
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image
import requests
from io import BytesIO
from pycocotools.coco import COCO
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from coco_91_to_80_cls import coco91_to_coco80_class, coco80_class
from tqdm import tqdm
import re 
import pandas as pd
from vllm import LLM, SamplingParams

def read_and_sort_ap(file_path):
    # 파일 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 유효한 데이터 라인 추출
    data_lines = lines[3:-1]  # 헤더와 마지막 라인 제외

    # 데이터 파싱
    categories = []
    ap_values = []
    for line in data_lines:
        parts = line.split('|')
        if len(parts) >= 7:
            categories.append(parts[1].strip() if parts[1].strip() != 'None' else 'background')
            ap_values.append(float(parts[2].strip()))
            categories.append(parts[3].strip() if parts[3].strip() != 'None' else 'background')
            ap_values.append(float(parts[4].strip()))
            categories.append(parts[5].strip() if parts[5].strip() != 'None' else 'background')
            ap_values.append(float(parts[6].strip()))

    # 데이터프레임 생성
    df = pd.DataFrame({'category': categories, 'AP': ap_values})

    # AP 성능이 낮은 순서대로 정렬
    df_sorted = df.sort_values(by='AP')

    return df_sorted
def get_lowest_ap_class_and_bbox(coco_classes, bboxes, sorted_ap):
    # 주어진 coco_classes 목록에서 AP가 가장 낮은 클래스 찾기
    ap_dict = dict(zip(sorted_ap['category'], sorted_ap['AP']))

    lowest_ap_index = min(range(len(coco_classes)), key=lambda i: ap_dict.get(coco_classes[i], float('inf')))
    lowest_ap_class = coco_classes[lowest_ap_index]
    lowest_ap_bbox = bboxes[lowest_ap_index]



    return lowest_ap_class, lowest_ap_bbox

def get_lowest_count_class_and_bbox(coco_classes, bboxes, sorted_class_counts):
    # 주어진 coco_classes 목록에서 가장 적은 개수의 클래스 찾기
    class_counts_dict = dict(sorted_class_counts)
    lowest_count_class = None
    lowest_count_bbox = None
    for cls_name, _ in sorted_class_counts:
        if cls_name in coco_classes:
            lowest_count_class = cls_name
            lowest_count_bbox = bboxes[coco_classes.index(cls_name)]
            break
    return lowest_count_class, lowest_count_bbox

def pad_collate_fn(batch):
    image, img_ids, bbox, cls, hard_class, prompt_base, prompt_loc, prompt_char, prompt_where = zip(*batch)

    # Bboxes와 classes를 리스트로 유지
    return list(image), list(img_ids), list(bbox), list(cls), list(hard_class), list(prompt_base), list(prompt_loc), list(prompt_char), list(prompt_where)

class COCODataset(Dataset):
    def __init__(self, img_dir, hard_cls_info, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.hard_cls_info = read_and_sort_ap(hard_cls_info)

        #self.ids = list(self.coco.imgs.keys())
        self.ids = self.coco.getImgIds()

        self.convert_91to80 = coco91_to_coco80_class()
        self.class_names = coco80_class()

        ## prompt
        #self.instruct_prompt='The contents of {} are the present object in the image. describe about provided object'
        self.question_base = 'describe about context' 
        self.question_loc = 'describe about context with location'
        self.question_char = 'describe about context with characteristics'
        self.question_where = 'describe about context with whole other objects'

        self.sorted_class_counts = self.get_class_counts()
        self.missing_count = 0

    def prompt_maker(self, context):
        prompt_base = f"USER: <image>\nContext:{context}\nQuestion:{self.question_base}\nASSISTANT:"
        prompt_loc = f"USER: <image>\nContext:{context}\nQuestion:{self.question_loc}\nASSISTANT:"
        prompt_char = f"USER: <image>\nContext:{context}\nQuestion:{self.question_char}\nASSISTANT:"
        prompt_where = f"USER: <image>\nContext:{context}\nQuestion:{self.question_where}\nASSISTANT:"
        return prompt_base, prompt_loc, prompt_char, prompt_where
    

    def get_class_counts(self):
        class_counts = {cat_id: 0 for cat_id in self.coco.getCatIds()}
        for ann in self.coco.anns.values():
            class_counts[ann['category_id']] += 1
        class_counts = sorted([(self.coco.cats[cat_id]['name'], count) for cat_id, count in class_counts.items()],
                              key=lambda x: x[1])
        return class_counts

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.ids[image_index], iscrowd=False)

        if len(annotations_ids) == 0:
            self.missing_count+=1
            return None, None
            #return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)

        boxes = []
        coco_classes = []
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            cls_n = self.convert_91to80[a['category_id']]
            if cls_n is None:
                continue

            cls_n = self.class_names[cls_n-1]
            boxes.append(a['bbox'])
            coco_classes.append(cls_n)

        return boxes, coco_classes

    def remove_special_characters(self,text, keep_characters=''):
        pattern = r'[^\w\s' + re.escape(keep_characters) + '()]|[\[\]\'\"]'
        return re.sub(pattern, '', text)
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        if os.path.isfile(path):
            image = Image.open(path).convert('RGB')
        else:
            image = False


        # Bbox 정보 로딩
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)
        bboxes, classes = self.load_annotations(idx)
        # get info for context
        ##### hard class
        hard_bbox, hard_class = bboxes, classes
        #rare_bbox, rare_class = bboxes, classes
        if bboxes==None:
            pass
        elif len(classes)>1:
            '''
            remain only hard class
            '''
            hard_class, hard_bbox = get_lowest_ap_class_and_bbox(classes, bboxes, self.hard_cls_info)
        hard_class = self.remove_special_characters(str(hard_class))

        ##### rare class
        #classes_set = set(classes)

        ## make context
        # Where is it located in the image? # Object 설명 ( 설명 방법 : 주위 환경에 대한 Context ) - 외부
        # What are its main characteristics (e.g., color, size, shape)? # Object 설명 ( 설명 방법 : Object의 상태 Context ) - 내부
        context =  str(hard_class)+': '+str(hard_bbox)
        prompt_base, prompt_loc, prompt_char, prompt_where = self.prompt_maker(context)
        '''
        print("*"*100)
        print("image:", image)
        print("img_id:", img_id)
        print("bboxes:", bboxes)
        print("classes:", classes)
        print("hard_class:", hard_class)
        print("prompt_base:", prompt_base)
        print("prompt_loc:", prompt_loc)
        print("prompt_char:", prompt_char)
        print("prompt_where:", prompt_where)
        '''
        return image, img_id, bboxes, classes, hard_class, prompt_base, prompt_loc, prompt_char, prompt_where

######## captioning model load ########
def load_captioning_model(model_id):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage=True, quantization_config = quantization_config)
    processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
    return processor, model


######## post processing ########
import torch

class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.output = None

    def hook_fn(self, module, input, output):
        self.output = output

    def close(self):
        self.hook.remove()
# hook for save embed

def remove_stopwords_en(text, extra_stopwords):
    # 단어 토큰화
    word_tokens = word_tokenize(text)

    # 영어 불용어 목록 가져오기
    stop_words = set(stopwords.words('english')+extra_stopwords)
    # 불용어 제거
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]

    # 결과를 문자열로 변환하여 반환
    return ' '.join(filtered_sentence)

def postprocessing(text, extra_stopwords):
    # "ASSISTANT: " 이후의 텍스트만 고려
    text = text.split("ASSISTANT: ")[-1].strip()

    #',', '.', '!', '?' 중 가장 먼저 나오는 문자의 위치 찾기
    end_indices = [text.find(char) for char in [',', '.', '!', '?'] if char in text]
    if end_indices:
        # 가장 먼저 나오는 문장 부호의 위치
        first_end = min(end_indices)
        # 문장 부호를 포함한 첫 문장 반환
        text = text[:first_end]
    text = remove_stopwords_en(text, extra_stopwords)
    return text

def captioning(processor, model, prompts, images, max_length, hard_classes, max_attempts=1):
    inputs = processor(prompts, images=images, return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].to('cuda')
    inputs['attention_mask'] = inputs['attention_mask'].to('cuda')
    inputs['pixel_values'] = inputs['pixel_values'].to('cuda')

    output_idx = model.generate(**inputs, max_length=max_length)
    output_texts = processor.batch_decode(output_idx, skip_special_tokens=True, temperature=0.0, num_beams=1)
    #processed_texts = postprocessing(output_texts[0], extra_stopwords)
    processed_texts = output_texts[0].split('ASSISTANT')[-1]
    if hard_classes[0] in processed_texts:
        return processed_texts, True #  when text is involve hard cls
    else:
        return processed_texts, False # when text is do not involve hard cls

import json
from pycocotools.coco import COCO
def save_result_see(args, results):
    new_txt_file = os.path.join(args.output_file, args.output_file+'.txt')
    with open(new_txt_file, 'a') as f:
        for x,y in results.items():
            sent = str(x)+':'+str(y)
            f.write(sent+'\n')

def save_result(args, results):
    # Load your existing COCO dataset
    coco = COCO(args.ann_file)
    # Iterate over each image ID in the COCO dataset
    for img_id in coco.imgs.keys():
        if img_id in results:
            # Update the COCO dataset with 'caption' field
            coco.imgs[img_id]['caption'] = results[img_id]
        else:
            # Handle cases where LLM did not produce an output for some images
            coco.imgs[img_id]['caption'] = "XX"  # Or any default value you prefer

    # Save the updated COCO dataset to a new JSON file
    new_json_file = os.path.join(args.output_file, args.output_file+'.json')

    print("save at :", new_json_file)
    if os.path.exists(new_json_file):
        os.remove(new_json_file)

    with open(new_json_file, 'w') as f:
        json.dump(coco.dataset, f)

    save_result_see(args, results)


def cm(processor, model, con3_prompt, images, hard_class, embed_last_hook, conv3_embed_last_p, img_ids, fail_t, pstage):
    con3_result, inc_hard_cls_c3 = captioning(processor, model, con3_prompt[0], images[0], 100, hard_class[0])
    if inc_hard_cls_c3 is True:
        conv3_embed_last_output = embed_last_hook.output
        conv3_embed_last_save_p = os.path.join(conv3_embed_last_p, str(img_ids[0])+".npy")
        np.save(conv3_embed_last_save_p, conv3_embed_last_output.squeeze().cpu().numpy())
    else:
        fail_t.write(str(img_ids[0])+"-"+pstage+":"+hard_class[0][0]+"->"+ con3_result)
    return con3_result, inc_hard_cls_c3

def save_embedding_avg_cls_max(feature):
    feat = feature[0]
    avg_pooled_feature = torch.mean(feat, dim=0)
    cls_feature = feat[0, :]  # (batch_size, 4096)
    max_pooled_feature = torch.max(feat, dim=0)[0]  # (batch_size, 4096)
    return avg_pooled_feature.cpu().numpy(), cls_feature.cpu().numpy(), max_pooled_feature.cpu().numpy()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--ann-file", type=str, required=True)
    parser.add_argument("--hard_cls_result", type=str, required=True)

    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--batch_n", type=int, default=1)

    #parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")    
    args = parser.parse_args()

    # 불용어 처리를 위한 NLTK 데이터 다운로드 
    nltk.download('punkt')
    nltk.download('stopwords')
    extra_stopwords = ['context','a','the','image','feature','features','depict','depicts']

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    D_HOOK = Hook(llm.llm_engine.model_executor.driver_worker.model_runner.model.language_model.norm)

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        #transforms.ToTensor(),
    ])
    dataset = COCODataset(img_dir=args.image_dir, hard_cls_info=args.hard_cls_result, ann_file=args.ann_file, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_n, shuffle=False, num_workers=16, collate_fn=pad_collate_fn) #collate_fn=lambda x: x)

    results = {} # save at result
    analysis = [0,0,0,0]

    f = open('save_result.txt', 'a')
    
    save_path = args.output_file
    save_path_1, save_path_2, save_path_3, save_path_4 = os.path.join(save_path, 'p1'), os.path.join(save_path, 'p2'), os.path.join(save_path, 'p3'), os.path.join(save_path, 'p4')
    save_path_1_cls, save_path_1_avg, save_path_1_max = os.path.join(save_path_1, "cls"), os.path.join(save_path_1, "avg"), os.path.join(save_path_1, "max")
    save_path_2_cls, save_path_2_avg, save_path_2_max = os.path.join(save_path_2, "cls"), os.path.join(save_path_2, "avg"), os.path.join(save_path_2, "max")
    save_path_3_cls, save_path_3_avg, save_path_3_max = os.path.join(save_path_3, "cls"), os.path.join(save_path_3, "avg"), os.path.join(save_path_3, "max")
    save_path_4_cls, save_path_4_avg, save_path_4_max = os.path.join(save_path_4, "cls"), os.path.join(save_path_4, "avg"), os.path.join(save_path_4, "max")

    for x in [save_path, save_path_1, save_path_2, save_path_3, save_path_4, save_path_1_cls, save_path_1_avg, save_path_1_max, save_path_2_cls, save_path_2_avg, save_path_2_max, save_path_3_cls, save_path_3_avg, save_path_3_max, save_path_4_cls, save_path_4_avg, save_path_4_max ]:
        os.makedirs(x, exist_ok=True)

    fail_t = open(os.path.join(save_path, "failure.txt"), 'a')

    csv_file_path = os.path.join(save_path, 'result.csv')

    # CSV 파일이 존재하지 않으면 새로 생성
    if not os.path.exists(csv_file_path):
        pd.DataFrame(columns=['img_ids', 'hard_class', 'bboxes', 'base_result', 'loc_result', 'char_result', 'where_result']).to_csv(csv_file_path, index=False)


    sample_count=0
    for images, image_ids, bboxes, cls, hard_class, prompt_base, prompt_loc, prompt_char, prompt_where in tqdm(data_loader):
        sample_count+=1
        #if sample_count!=19 or sample_count!=21:
        #    continue
        if bboxes[0] is None or images[0] is False: # if image does not contain any annotations 
            continue
        images = images[0]
        prompt_1 = [{"prompt": prompt_base[0], "multi_modal_data": {"image": images}}]
        prompt_2 = [{"prompt": prompt_loc[0], "multi_modal_data": {"image": images}}]
        prompt_3 = [{"prompt": prompt_char[0], "multi_modal_data": {"image": images}}]
        prompt_4 = [{"prompt": prompt_where[0], "multi_modal_data": {"image": images}}]


        outputs_1 = llm.generate(prompt_1, sampling_params=sampling_params)
        D_Feature1 = D_HOOK.output
        avg_embed_1, cls_embed_1, max_embed_1 = save_embedding_avg_cls_max(D_Feature1)
        np.save(os.path.join(save_path_1_avg, str(image_ids[0])+'.npy'), avg_embed_1)
        np.save(os.path.join(save_path_1_cls, str(image_ids[0])+'.npy'), cls_embed_1)
        np.save(os.path.join(save_path_1_max, str(image_ids[0])+'.npy'), max_embed_1)

        outputs_2 = llm.generate(prompt_2, sampling_params=sampling_params)
        D_Feature2 = D_HOOK.output
        avg_embed_2, cls_embed_2, max_embed_2 = save_embedding_avg_cls_max(D_Feature2)
        np.save(os.path.join(save_path_2_avg, str(image_ids[0])+'.npy'), avg_embed_2)
        np.save(os.path.join(save_path_2_cls, str(image_ids[0])+'.npy'), cls_embed_2)
        np.save(os.path.join(save_path_2_max, str(image_ids[0])+'.npy'), max_embed_2)

        outputs_3 = llm.generate(prompt_3, sampling_params=sampling_params)
        D_Feature3 = D_HOOK.output
        avg_embed_3, cls_embed_3, max_embed_3 = save_embedding_avg_cls_max(D_Feature3)
        np.save(os.path.join(save_path_3_avg, str(image_ids[0])+'.npy'), avg_embed_3)
        np.save(os.path.join(save_path_3_cls, str(image_ids[0])+'.npy'), cls_embed_3)
        np.save(os.path.join(save_path_3_max, str(image_ids[0])+'.npy'), max_embed_3)

        outputs_4 = llm.generate(prompt_4, sampling_params=sampling_params)
        D_Feature4 = D_HOOK.output
        avg_embed_4, cls_embed_4, max_embed_4 = save_embedding_avg_cls_max(D_Feature4)
        np.save(os.path.join(save_path_4_avg, str(image_ids[0])+'.npy'), avg_embed_4)
        np.save(os.path.join(save_path_4_cls, str(image_ids[0])+'.npy'), cls_embed_4)
        np.save(os.path.join(save_path_4_max, str(image_ids[0])+'.npy'), max_embed_4)


