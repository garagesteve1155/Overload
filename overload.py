#!/usr/bin/env python
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.utils import logging
logging.set_verbosity_error()
from safetensors.torch import load_file
import gc
import re
import sys
import psutil
import time
from contextlib import contextmanager
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR   = "./mistral_7b_instruct"
MODEL_INDEX = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
TOTAL_LAYERS = 32

DTYPE_MAP = {
    "F32":  torch.float32,
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "I64":  torch.int64,
    "I32":  torch.int32,
    "BOOL": torch.bool,
    "U8":   torch.uint8,
}

NUMPY_DTYPE_MAP = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64:   np.int64,
    torch.int32:   np.int32,
    torch.bool:    np.bool_,
    torch.uint8:   np.uint8,
}

def get_dtype(dtype_str):
    return DTYPE_MAP[dtype_str]

def load_tensor_from_shard(model_dir, filename, tensor_name):
    full_path = os.path.join(model_dir, filename)

    with open(full_path, "rb") as f:
        header_size  = int.from_bytes(f.read(8), "little")
        tensor_header = json.loads(f.read(header_size).decode("utf-8"))
        info   = tensor_header[tensor_name]
        dtype  = get_dtype(info["dtype"])
        shape  = info["shape"]
        start, end = info["data_offsets"]

        f.seek(8 + header_size + start)
        data = f.read(end - start)

    if dtype == torch.bfloat16:
        arr_u8  = np.frombuffer(data, dtype=np.uint8)
        arr_u16 = arr_u8.view(np.uint16)
        arr_i16 = arr_u16.view(np.int16)
        t_int16 = torch.from_numpy(arr_i16)
        tensor  = t_int16.view(torch.bfloat16).reshape(shape)
    else:
        np_dtype = NUMPY_DTYPE_MAP[dtype]
        tensor   = torch.from_numpy(np.frombuffer(data, dtype=np_dtype)).reshape(shape)
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
        self.config    = AutoConfig.from_pretrained(CONFIG_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

        with open(MODEL_INDEX, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        self.tensor_meta = defaultdict(dict)
        for tensor_name, filename in index_data["weight_map"].items():
            m = re.match(r"model\.layers\.(\d+)\.(.+)", tensor_name)
            if m:
                layer_idx = int(m.group(1))
                subkey    = m.group(2)
                self.tensor_meta[layer_idx][subkey] = {
                    "filename": filename,
                    "tensor_name": tensor_name
                }
            else:
                self.tensor_meta["global"][tensor_name] = {
                    "filename": filename,
                    "tensor_name": tensor_name
                }

        self.device = torch.device("cpu")

        self.embedding = None
        self.ln_f      = None
        self.lm_head   = None

        self.adapter_state = None
        if adapter_path is not None:
            print(f"[INFO] Loading adapter from: {adapter_path}")
            self.adapter_state = load_file(adapter_path, device="cpu")


    def prepare_inputs(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    def embed(self, input_ids):
        return self.embedding(input_ids)

    def cat_tokens(self, ids_a, ids_b):
        return torch.cat([ids_a, ids_b], dim=1)

    def build_mask(self, cur_len, past_len, dtype):
        return build_attention_mask(cur_len, past_len, self.device, dtype)

    def position_ids(self, length):
        return torch.arange(0, length, device=self.device).unsqueeze(0)

    def norm_and_logits(self, hidden_state):
        normed  = self.ln_f(hidden_state)
        logits  = self.lm_head(normed[:, -1, :])
        return logits

    # -------------- NEW: torch‑free context for __main__ --------------- #
    @contextmanager
    def no_grad(self):
        with torch.no_grad():
            yield
    # ------------------------------------------------------------------ #

    def load_embedding(self):
        for tensor_name, meta in self.tensor_meta["global"].items():
            if "embed" in tensor_name and "weight" in tensor_name:
                tensor = load_tensor_from_shard(
                    self.model_dir, meta["filename"], meta["tensor_name"]
                )
                self.embedding = torch.nn.Embedding(
                    tensor.size(0), tensor.size(1)
                ).to(self.device)
                self.embedding.weight.data.copy_(tensor)
                self.embedding.eval()
                break
    def load_needed(self):
        """
        Load the embedding, LM head, and final layer norm in one call.
        """
        self.load_embedding()
        self.load_lm_head()
        self.load_ln_f()

    def load_layers(self, start_idx, end_idx, device=None):  # <--- add device arg here
        group_layers     = []
        layer_tensor_map = defaultdict(dict)

        tensor_list = []
        for i in range(start_idx, end_idx):
            if i not in self.tensor_meta:
                continue
            for subkey, meta in self.tensor_meta[i].items():
                tensor_list.append((i, subkey, meta["filename"], meta["tensor_name"]))

        def load_tensor_wrapper(args):
            i, subkey, filename, tensor_name = args
            tensor = load_tensor_from_shard(self.model_dir, filename, tensor_name)
            return i, subkey, tensor

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            for i, subkey, tensor in ex.map(load_tensor_wrapper, tensor_list):
                layer_tensor_map[i][subkey] = tensor

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

            # Use custom device if provided
            target_device = device if device else self.device
            block.to(target_device)
            block.eval()
            block.device = target_device
            group_layers.append(block)

        return group_layers

        
    def load_lm_head(self):
        meta = next(
            v for k, v in self.tensor_meta["global"].items() if "lm_head.weight" in k
        )
        tensor = load_tensor_from_shard(
            self.model_dir, meta["filename"], meta["tensor_name"]
        )
        self.lm_head = torch.nn.Linear(
            tensor.size(1), tensor.size(0), bias=False
        ).to(self.device)
        self.lm_head.weight.data.copy_(tensor)
        self.lm_head.eval()

    def load_ln_f(self):
        meta = self.tensor_meta["global"]["model.norm.weight"]
        tensor = load_tensor_from_shard(
            self.model_dir, meta["filename"], meta["tensor_name"]
        )
        self.ln_f = MistralRMSNorm(self.config.hidden_size, eps=1e-6).to(self.device)
        self.ln_f.weight.data.copy_(tensor)
        self.ln_f.eval()
    def encode(self, input_ids):
        """
        Given input_ids, run the embedding and build the attention mask
        and position ids in one shot.
        Returns (hidden_state, attn_mask, position_ids).
        """
        cur_len      = input_ids.size(1)
        hidden_state = self.embed(input_ids)
        attn_mask    = self.build_mask(cur_len, 0, hidden_state.dtype)
        position_ids = self.position_ids(cur_len)
        return hidden_state, attn_mask, position_ids
    def generate_token(self, hidden_state, input_ids, step):
        """
        Given the current hidden_state and input_ids, run norm→logits→sampling,
        time it, advance input_ids, and return (new_input_ids, partial_output).
        Prints the elapsed time; main is responsible for printing the partial.
        """
        logits     = self.norm_and_logits(hidden_state)
        next_token = safe_sample(logits)
        input_ids = self.cat_tokens(input_ids, next_token)
        decoded   = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        partial   = (
            decoded.split("[/INST]")[-1].strip()
            if "[/INST]" in decoded
            else decoded
        )
        return input_ids, next_token, partial



#### EXAMPLE USAGE ####
if __name__ == "__main__":
    adapter_path = None
    runner = MistralLayerwiseRunner(MODEL_DIR, adapter_path)
    gpu_available = torch.cuda.is_available()
    gpu_device = torch.device("cuda") if gpu_available else None
    cpu_device = torch.device("cpu")
    if gpu_available:
        gpu_safety_buffer = float(input("How much GPU VRAM to leave open (In GB. Enter 0 to use whole GPU): "))

    runner.device = cpu_device
    runner.load_needed()

    eos_token_id = runner.tokenizer.convert_tokens_to_ids("</s>")
    total_ram = psutil.virtual_memory().total / (1024**3)
    if total_ram > 4:
        safety_buffer = float(input("How much RAM to leave open (Used for overload. 4GB Minimum): "))
    else:
        safety_buffer = 4
        print("Entering Overload Mode for all layers")
    print(f"[INFO] Measuring RAM usage per layer...")

    gpu_layers = []
    cpu_layers = []
    max_i = 0

    for i in range(TOTAL_LAYERS):
        try:
            if gpu_available:
                torch.cuda.empty_cache()
                mem_free, _ = torch.cuda.mem_get_info()
                gpu_mem_free_gb = mem_free / (1024**3)
                if gpu_mem_free_gb < gpu_safety_buffer:
                    raise RuntimeError("Low GPU memory")
                layer = runner.load_layers(i, i + 1, device=gpu_device)[0]
                gpu_layers.append(layer)
                print(f"[GPU] Layer {i} loaded.")
                max_i += 1
                continue
        except:
            print(f"[GPU] Cannot load layer {i}, switching to CPU.")
        
        if psutil.virtual_memory().available / (1024**3) < safety_buffer:
            print(f"[CPU] Not enough RAM at layer {i}, stopping preload.")
            break
        layer = runner.load_layers(i, i + 1, device=cpu_device)[0]
        cpu_layers.append(layer)
        print(f"[CPU] Layer {i} loaded.")
        max_i += 1

    preloaded_layers = gpu_layers + cpu_layers




    

    while True:
        max_new_tokens = 2048
        bar_len = 20
        past_cache = {}
        user_prompt = input("Prompt: ")
        print("\nGenerating. Please wait...")
        prompt = f"<s>[INST] {user_prompt} [/INST] "
        input_ids = runner.prepare_inputs(prompt)

        for step in range(max_new_tokens):
            hidden_state, attn_mask, position_ids = runner.encode(input_ids)
            token_start = time.time()
            with runner.no_grad():
                if preloaded_layers:
                    # Process preloaded layers first
                    for i, layer in enumerate(preloaded_layers):
                        start_time = time.time()

                        out = layer(
                            hidden_state.to(layer.device),
                            attention_mask=attn_mask.to(layer.device),
                            position_ids=position_ids.to(layer.device),
                            past_key_value=past_cache.get(i),
                            use_cache=False,
                        )

                        hidden_state = out[0].detach().to(cpu_device)
                        del out
                        if layer.device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        

                # Dynamically load remaining layers individually
                for i in range(max_i, TOTAL_LAYERS):
                    dynamic_layer = runner.load_layers(i, i + 1)[0]
                    out = dynamic_layer(
                        hidden_state,
                        attention_mask=attn_mask,
                        position_ids=position_ids,
                        past_key_value=past_cache.get(i),
                        use_cache=False,
                    )
                    hidden_state = out[0].detach()
                    del out, dynamic_layer
                    gc.collect()
                    if not preloaded_layers and i == 0 and step == 0:
                        print("Well, looks like it's actually working, even on this computer.\nWild, man...\nThis is going to take a while, though. Please keep waiting...")

            input_ids, next_token, partial_out = runner.generate_token(hidden_state, input_ids, step)
            print(f"\n[Partial Output]: {partial_out}")
            print("Token generated in "+ str(time.time()-token_start)+" seconds")
            if next_token.item() == eos_token_id:
                print("[INFO] Stop token </s> encountered.")
                break

        final_text = runner.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        ).split("[/INST]")[1]
        print("\n\nPrompt: "+user_prompt)
        print("\nResponse:", final_text)
