# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
os.environ.setdefault("PT_HPUGRAPH_DISABLE_TENSOR_CACHE", "1")
import copy
import time
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()

import habana_frameworks.torch.core as htcore

device = "hpu"
torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

# specify the path to the model
model_path = "/data/janus-pro-7b"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).to(device).eval()

vl_gpt.language_model = wrap_in_hpu_graph(vl_gpt.language_model)

generation_config = copy.deepcopy(vl_gpt.language_model.generation_config)
generation_config.max_new_tokens = 512
generation_config.use_cache = True
generation_config.static_shapes = True
generation_config.reuse_cache = False # Do not change it to True
generation_config.do_sample = False
generation_config.num_beams = 1
generation_config.top_k = 1
generation_config.num_return_sequences = 1
generation_config.trim_logits = True
generation_config.attn_softmax_bf16 = False
generation_config.limit_hpu_graphs = True
generation_config.use_flash_attention = True
generation_config.flash_attention_recompute = False
generation_config.flash_attention_causal_mask = False
generation_config.flash_attention_fast_softmax = False
generation_config.trust_remote_code = True
generation_config.valid_sequence_lengths = None


conversation1 = [
    {
        "role": "User",
        "content": "<image_placeholder>\n请描述这幅图片。",
        "images": ["./images/000055648.jpg"],
    },
    {"role": "Assistant", "content": ""},
]

# load images and prepare for inputs
pil_images1 = load_pil_images(conversation1)
prepare_inputs1 = vl_chat_processor(
    conversations=conversation1, images=pil_images1, force_batchify=True
).to(device)

conversation2 = [
    {
        "role": "User",
        "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\n请你描述这两幅图>片。",
        "images": [
            "./images/Beijing.jpeg",
            "./images/Chongqing.jpeg"
        ]
    },
    {"role": "Assistant", "content": ""},
]

# load images and prepare for inputs
pil_images2 = load_pil_images(conversation2)
prepare_inputs2 = vl_chat_processor(
    conversations=conversation2, images=pil_images2, force_batchify=True
).to(device)


loop = 10
for i in range(loop):
    print("Index = " + str(i))
    t1 = time.time()
    # # run image encoder to get the image embeddings
    if i % 2 == 0:
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs1)
        attention_mask = prepare_inputs1.attention_mask
    else:
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs2)
        attention_mask = prepare_inputs2.attention_mask

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        lazy_mode=True,
        generation_config=generation_config,
        hpu_graphs=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    t2 = time.time()
    if i % 2 == 0:
        print(f"{prepare_inputs1['sft_format'][0]}", answer)
    else:
        print(f"{prepare_inputs2['sft_format'][0]}", answer)
    print()
    print("-----------------------------------")
    print("IMG2TXT Infer Latency: {:3f} sec".format(t2-t1))
    print("-----------------------------------")
    print()
