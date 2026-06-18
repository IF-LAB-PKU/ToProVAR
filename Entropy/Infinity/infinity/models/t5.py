import re
import torch
import os
import traceback
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, T5EncoderModel

import ftfy
import html
from bs4 import BeautifulSoup
import urllib.parse as ul


class T5Embedder:
    
    available_models = ['t5-v1_1-xxl']
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa
    
    def __init__(self, device, dir_or_name='t5-v1_1-xxl', *, local_cache=False, cache_dir=None, hf_token=None, use_text_preprocessing=True,
                 t5_model_kwargs=None, torch_dtype=torch.bfloat16, use_offload_folder=None, model_max_length=512, padding="max_length", clean_caption_func_name="clean_caption"):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        if t5_model_kwargs is None:
            t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': self.torch_dtype}
            if use_offload_folder is not None:
                t5_model_kwargs['offload_folder'] = use_offload_folder
                t5_model_kwargs['device_map'] = {
                    'shared': self.device,
                    'encoder.embed_tokens': self.device,
                    'encoder.block.0': self.device,
                    'encoder.block.1': self.device,
                    'encoder.block.2': self.device,
                    'encoder.block.3': self.device,
                    'encoder.block.4': self.device,
                    'encoder.block.5': self.device,
                    'encoder.block.6': self.device,
                    'encoder.block.7': self.device,
                    'encoder.block.8': self.device,
                    'encoder.block.9': self.device,
                    'encoder.block.10': self.device,
                    'encoder.block.11': self.device,
                    'encoder.block.12': 'disk',
                    'encoder.block.13': 'disk',
                    'encoder.block.14': 'disk',
                    'encoder.block.15': 'disk',
                    'encoder.block.16': 'disk',
                    'encoder.block.17': 'disk',
                    'encoder.block.18': 'disk',
                    'encoder.block.19': 'disk',
                    'encoder.block.20': 'disk',
                    'encoder.block.21': 'disk',
                    'encoder.block.22': 'disk',
                    'encoder.block.23': 'disk',
                    'encoder.final_layer_norm': 'disk',
                    'encoder.dropout': 'disk',
                }
            else:
                t5_model_kwargs['device_map'] = {'shared': self.device, 'encoder': self.device}
        
        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        tokenizer_path, path = dir_or_name, dir_or_name
        if local_cache:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            tokenizer_path, path = cache_dir, cache_dir
        elif dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
                'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
            ]:
                hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path, path = cache_dir, cache_dir
        else:
            cache_dir = os.path.join(self.cache_dir, 't5-v1_1-xxl')
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
            ]:
                hf_hub_download(repo_id='DeepFloyd/t5-v1_1-xxl', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path = cache_dir
        
        print(f"Loading T5 from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        self.model_max_length = model_max_length
        self.padding = padding
        self.clean_caption_func = self.__getattribute__(clean_caption_func_name)
    
    @torch.no_grad()
    def get_text_embeddings(self, texts):
        import time
        start_time = time.time()
        
        texts = [self.text_preprocessing(text) for text in texts]
        
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding=self.padding,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids'].to(self.device)
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=text_tokens_and_mask['input_ids'],
                attention_mask=text_tokens_and_mask['attention_mask'],
            )['last_hidden_state'].detach()
        
        return text_encoder_embs, text_tokens_and_mask['attention_mask'], text_tokens_and_mask['input_ids'], texts
    
    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            try:
                # The exact text cleaning as was in the training stage:
                text = self.clean_caption_func(text)
                text = self.clean_caption_func(text)
                return text
            except Exception as e:
                print(f"Error in text preprocessing: {e} with text: {text}")
                print(traceback.format_exc())
                return text
        else:
            return text.lower().strip()
    
    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
    
    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        # html:
        try:
            caption = BeautifulSoup(caption, features='html.parser').text
        except Exception as e:
            print(f"Error parsing caption:{caption} with html.parser: {e}")
        
        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)
        
        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################
        
        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)
        
        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)
        
        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)
        
        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)
        
        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)
        
        # \n
        caption = re.sub(r'\\n', ' ', caption)
        
        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)
        
        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""
        
        caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "
        
        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)
        
        caption = self.basic_clean(caption)
        
        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231
        
        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)
        
        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...
        
        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)
        
        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)
        
        caption.strip()
        
        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)
        
        return caption.strip()
    
    
    def clean_caption_simplify(self, caption):
        # Convert caption to string.
        caption = str(caption)
        
        # Decode URL-encoded text.
        caption = ul.unquote_plus(caption)
        
        # Trim whitespace and lowercase.
        caption = caption.strip().lower()
        
        # Normalize person placeholders.
        caption = re.sub('<person>', 'person', caption)
        
        # Remove URLs.
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)  # matches URLs that begin with http:// or https://
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            '', caption)  # matches URLs that begin with www.
        
        # Parse HTML and remove tags.
        caption = BeautifulSoup(caption, features='html.parser').text
        
        # Remove @nickname tags.
        caption = re.sub(r'@[\w\d]+\b', '', caption)
        
        # Remove CJK-related Unicode ranges.
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)  # CJK strokes
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)  # Katakana phonetic extensions
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)  # enclosed CJK letters and months
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)  # CJK compatibility
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)  # CJK Unified Ideographs Extension A
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)  # Yijing hexagram symbols
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)  # CJK Unified Ideographs
        
        # Normalize dash characters to "-".
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            '-', caption)  # matches common Unicode dash characters
        
        # Normalize quote characters.
        caption = re.sub(r'[`´«»“”¨]', '"', caption)  # standard double quotes
        caption = re.sub(r'[‘’]', "'", caption)  # standard single quotes
        
        # Remove common HTML entities.
        caption = re.sub(r'&quot;?', '', caption)
        caption = re.sub(r'&amp', '', caption)
        
        # Remove IPv4 addresses.
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)
        
        # Remove article ID-like suffixes.
        caption = re.sub(r'\d:\d\d\s+$', '', caption)  # e.g. "1:23 "
        
        # Remove escaped newlines.
        caption = re.sub(r'\\n', ' ', caption)
        
        # Remove media and document filenames.
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)
        
        # Collapse repeated quotes and periods.
        caption = re.sub(r'[\"\']{2,}', r'"', caption)
        caption = re.sub(r'[\.]{2,}', r' ', caption)
        
        # Remove invalid punctuation and isolated periods.
        caption = re.sub(self.bad_punct_regex, r' ', caption)
        caption = re.sub(r'\s+\.\s+', r' ', caption)
        
        # Replace excessive dashes or underscores.
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)
        
        # Apply basic text cleanup.
        caption = self.basic_clean(caption)
        
        # Remove rectangular dimension markers.
        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)
        
        # Normalize whitespace and punctuation.
        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)
        
        # Remove leading and trailing punctuation artifacts.
        caption.strip()
        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)
        
        return caption.strip()
