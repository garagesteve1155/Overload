#!/usr/bin/env python

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.utils import logging
logging.set_verbosity_error()
from safetensors.torch import load_file
import base64
import io
import gc
import re
import sys
import time
import psutil
from contextlib import contextmanager
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

@contextmanager
def timer(label):
    start = time.time()
    yield
    end = time.time()
    print(f"[⏱️ {label}] {end - start:.4f} sec")


MODEL_DIR = "./mistral_7b_instruct"
MODEL_INDEX = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
TOTAL_LAYERS = 32

DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
    "U8": torch.uint8,
}

NUMPY_DTYPE_MAP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
}

def get_dtype(dtype_str):
    return DTYPE_MAP[dtype_str]

def load_tensor_from_shard(model_dir, filename, tensor_name):
    full_path = os.path.join(model_dir, filename)

    with open(full_path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        tensor_header = json.loads(f.read(header_size).decode("utf-8"))
        info = tensor_header[tensor_name]
        dtype = get_dtype(info["dtype"])
        shape = info["shape"]
        start, end = info["data_offsets"]

        f.seek(8 + header_size + start)
        data = f.read(end - start)

    if dtype == torch.bfloat16:
        arr_u8 = np.frombuffer(data, dtype=np.uint8)
        arr_u16 = arr_u8.view(np.uint16)
        arr_i16 = arr_u16.view(np.int16)
        t_int16 = torch.from_numpy(arr_i16)
        tensor = t_int16.view(torch.bfloat16).reshape(shape)
    else:
        np_dtype = NUMPY_DTYPE_MAP[dtype]
        tensor = torch.from_numpy(np.frombuffer(data, dtype=np_dtype)).reshape(shape)
    del data

    tensor = tensor.to(torch.float16)
    return tensor

def build_attention_mask(current_len, past_len, device, dtype):
    total_len = past_len + current_len
    mask = torch.full((1, 1, current_len, total_len), float("-inf"), dtype=dtype, device=device)
    return torch.triu(mask, diagonal=1 + past_len)

def safe_sample(logits):
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, 10)
    return torch.multinomial(probs, num_samples=1)


class MistralLayerwiseRunner:
    def __init__(self, model_dir, adapter_path=None):
        self.model_dir = model_dir
        self.config = AutoConfig.from_pretrained(CONFIG_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        with open(MODEL_INDEX, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.tensor_meta = defaultdict(dict)
        for tensor_name, filename in index_data["weight_map"].items():
            # Extract layer index if it exists
            match = re.match(r"model\.layers\.(\d+)\.(.+)", tensor_name)
            if match:
                layer_idx = int(match.group(1))
                subkey = match.group(2)
                self.tensor_meta[layer_idx][subkey] = {
                    "filename": filename,
                    "tensor_name": tensor_name
                }
            else:
                # Handle embedding, norm, lm_head
                self.tensor_meta["global"][tensor_name] = {
                    "filename": filename,
                    "tensor_name": tensor_name
                }

        self.index_data = index_data  # keep the raw JSON around if needed

        self.device = torch.device("cpu")


        self.embedding = None
        self.ln_f = None
        self.lm_head = None

        self.adapter_state = None
        if adapter_path is not None:
            print(f"[INFO] Loading adapter from: {adapter_path}")
            from safetensors.torch import load_file
            self.adapter_state = load_file(adapter_path, device="cpu")

    def load_embedding(self):
        for tensor_name, meta in self.tensor_meta["global"].items():
            if "embed" in tensor_name and "weight" in tensor_name:
                tensor = load_tensor_from_shard(self.model_dir, meta["filename"], meta["tensor_name"])
                self.embedding = torch.nn.Embedding(tensor.size(0), tensor.size(1)).to(self.device)
                self.embedding.weight.data.copy_(tensor)
                self.embedding.eval()
                break


    def load_layers(self, start_idx, end_idx):
        group_layers = []
        layer_tensor_map = defaultdict(dict)

        # Get tensor list for this group
        tensor_list = []
        for i in range(start_idx, end_idx):
            if i not in self.tensor_meta:
                continue
            for subkey, meta in self.tensor_meta[i].items():
                tensor_list.append((i, subkey, meta["filename"], meta["tensor_name"]))

        # Load in parallel
        def load_tensor_wrapper(args):
            i, subkey, filename, tensor_name = args
            tensor = load_tensor_from_shard(self.model_dir, filename, tensor_name)
            return i, subkey, tensor

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            for i, subkey, tensor in executor.map(load_tensor_wrapper, tensor_list):
                layer_tensor_map[i][subkey] = tensor

        # Build layers
        for i in range(start_idx, end_idx):
            layer_state = layer_tensor_map[i]

            if self.adapter_state is not None:
                adapter_prefix = f"model.layers.{i}."
                for a_key, a_tensor in self.adapter_state.items():
                    if a_key.startswith(adapter_prefix):
                        adapter_subkey = a_key.split(".", 3)[-1]
                        if adapter_subkey in layer_state:
                            print(f"[DEBUG] Merging adapter update for layer {i} key {adapter_subkey}")
                            layer_state[adapter_subkey] += a_tensor.to(torch.float32)

            block = MistralDecoderLayer(self.config, layer_idx=i)
            block.load_state_dict(layer_state, strict=False)
            block.to(self.device)
            block.eval()
            group_layers.append(block)

        return group_layers



    def load_lm_head(self):
        meta = next(v for k, v in self.tensor_meta["global"].items() if "lm_head.weight" in k)
        tensor = load_tensor_from_shard(self.model_dir, meta["filename"], meta["tensor_name"])
        self.lm_head = torch.nn.Linear(tensor.size(1), tensor.size(0), bias=False).to(self.device)
        self.lm_head.weight.data.copy_(tensor)
        self.lm_head.eval()


    def load_ln_f(self):
        meta = self.tensor_meta["global"]["model.norm.weight"]
        tensor = load_tensor_from_shard(self.model_dir, meta["filename"], meta["tensor_name"])
        self.ln_f = MistralRMSNorm(self.config.hidden_size, eps=1e-6).to(self.device)
        self.ln_f.weight.data.copy_(tensor)
        self.ln_f.eval()


    def run(self, prompt: str, max_new_tokens=2):
        

        GROUP_SIZE = 1

        self.load_embedding()
        self.load_lm_head()
        self.load_ln_f()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.size(1)
        past_cache = {}

        hidden_state = self.embedding(input_ids)
        attn_mask = build_attention_mask(prompt_length, 0, self.device, hidden_state.dtype)
        position_ids = torch.arange(0, prompt_length, device=self.device).unsqueeze(0)
        with torch.no_grad():
            for group_start in range(0, TOTAL_LAYERS, GROUP_SIZE):
                group_end = min(group_start + GROUP_SIZE, TOTAL_LAYERS)
                group_layers = self.load_layers(group_start, group_end)

                for i, layer in enumerate(group_layers, start=group_start):
                    out = layer(
                        hidden_state,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=None,
                        use_cache=False
                    )
                    hidden_state = out[0].detach()
                    del out
                    del layer

                    progress = int((i + 1) / TOTAL_LAYERS * 100)
                    bar_len = 20
                    filled = int(progress / 100 * bar_len)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(f"\rPrompt Processing: [{bar}] {progress:3d}%")
                    sys.stdout.flush()
                del group_layers
                gc.collect()
        print() 

        eos_token_id = self.tokenizer.convert_tokens_to_ids("</s>")

        
        for step in range(max_new_tokens):
            token_start_time = time.time()

            current_length = input_ids.size(1)
            hidden_state = self.embedding(input_ids)
            attn_mask = build_attention_mask(current_length, 0, self.device, hidden_state.dtype)
            position_ids = torch.arange(0, current_length, device=self.device).unsqueeze(0)
            max_mem = 0.0
            cpu_mem = 0.0
            with torch.no_grad():
                for group_start in range(0, TOTAL_LAYERS, GROUP_SIZE):
                    group_end = min(group_start + GROUP_SIZE, TOTAL_LAYERS)
                    group_layers = self.load_layers(group_start, group_end)
                    for i, layer in enumerate(group_layers, start=group_start):
                        out = layer(
                            hidden_state,
                            attention_mask=attn_mask,
                            position_ids=position_ids,
                            past_key_value=past_cache.get(i),
                            use_cache=False
                        )
                        hidden_state = out[0].detach()
                        del out
                        del layer
                        past_cache = {}
                        progress = int((i + 1) / TOTAL_LAYERS * 100)
                        bar_len = 20
                        filled = int(progress / 100 * bar_len)
                        bar = "█" * filled + "-" * (bar_len - filled)
                        sys.stdout.write(f"\r[Token {step + 1}] Layer Progress: [{bar}] {progress:3d}%")
                        sys.stdout.flush()
           
                    process = psutil.Process(os.getpid())
                    cpu_mem_new = process.memory_info().rss / (1024**2)
                    if cpu_mem_new > cpu_mem:
                        cpu_mem = cpu_mem_new
                    del group_layers
                    gc.collect()
         
            print(f"\n[Token {step + 1}] RAM usage during pass: {cpu_mem:.2f} MB")

            normed = self.ln_f(hidden_state)
            logits = self.lm_head(normed[:, -1, :])
            next_token = safe_sample(logits)

            token_end_time = time.time()
            elapsed_time = token_end_time - token_start_time
            print(f"\n[Token {step + 1}] generation took {elapsed_time:.4f} seconds")

            input_ids = torch.cat([input_ids, next_token], dim=1)
            decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            partial_text = decoded.split("[/INST]")[-1].strip() if "[/INST]" in decoded else decoded
            print(f"[Partial Output]: {partial_text}")

            if next_token.item() == eos_token_id:
                print("[INFO] Stop token </s> encountered.")
                break

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)



if __name__ == "__main__":
    adapter_path = None
    runner = MistralLayerwiseRunner(MODEL_DIR, adapter_path=adapter_path)
    while True:
        prompt = input("Prompt: ")
        result = runner.run(f"<s>[INST] {prompt} [/INST] ", max_new_tokens=2048)
        print("\nResponse: ", result)
