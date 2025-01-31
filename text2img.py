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
import time
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

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

vl_gpt.language_model.model = wrap_in_hpu_graph(vl_gpt.language_model.model)
vl_gpt.gen_aligner = wrap_in_hpu_graph(vl_gpt.gen_aligner)
vl_gpt.gen_vision_model.decoder = wrap_in_hpu_graph(vl_gpt.gen_vision_model.decoder)

conversation1 = [
    {
        "role": "User",
        "content": "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format1 = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation1,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt1 = sft_format1 + vl_chat_processor.image_start_tag

conversation2 = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format2 = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation2,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt2 = sft_format2 + vl_chat_processor.image_start_tag

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    folder_idx: int = 0,
):
    device = "hpu"

    t1 = time.time()

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_len = len(input_ids)
    actual_len = input_len + image_token_num_per_image
    total_len = int(np.ceil(actual_len / 256)) * 256
    padding_len = total_len - input_len
    input_ids = input_ids + [vl_chat_processor.pad_id] * padding_len
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, total_len), dtype=torch.int, device="cpu")
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:(input_len - 1)] = vl_chat_processor.pad_id
    tokens = tokens.to(device)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    decoder_attention_mask = torch.zeros([parallel_size*2, total_len], dtype=torch.long, device=device)
    mask_index = torch.arange(0, input_len, device=device)
    decoder_attention_mask.index_fill_(1, mask_index, 1)

    for i in range(image_token_num_per_image):
        if i == 0:
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                use_flash_attention=True,
                flash_attention_causal_mask=True,
                lazy_mode=True,
            )
        else:
            token_idx = torch.tensor([input_len + i], dtype=torch.long, device=device)
            position_ids = torch.tensor([input_len - 1 + i], dtype=torch.long, device=device).unsqueeze(0)
            decoder_attention_mask.index_fill_(1, position_ids.squeeze(0), 1)
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                past_key_values=outputs.past_key_values,
                use_cache=True,
                token_idx=token_idx,
                use_flash_attention=True,
                flash_attention_causal_mask=False,
                lazy_mode=True,
            )

        hidden_states = outputs.last_hidden_state
        
        if i == 0:
            logits = mmgpt.gen_head(hidden_states[:, input_len - 1, :])
        else:
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        #generated_tokens[:, i] = next_token.squeeze(dim=-1)
        copy_idx = torch.tensor([i], device=device)
        generated_tokens.index_copy_(1, copy_idx, next_token)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    htcore.mark_step()
    t2 = time.time()

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    htcore.mark_step()
    t3 = time.time()

    saved_folder = 'generated_samples_' + str(folder_idx)
    os.makedirs(saved_folder, exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join(saved_folder, "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)
    t4 = time.time()
    print("LLM Infer Latency: {:3f} sec".format(t2-t1))
    print("VisionDec Latency: {:3f} sec".format(t3-t2))
    print("SaveImage Latency:  {:3f} sec".format(t4-t3))

loop = 10
for i in range(loop):
    print("Index = " + str(i))
    generate(
        vl_gpt,
        vl_chat_processor,
        prompt1 if i % 2 == 0 else prompt2,
        folder_idx=i,
    )
    print()
