import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append('../../code/llm/LLaVA')
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
model_path = "../../code/llm/LLaVA-13b"
# NOTE:"requires transformer package version==4.37.1"
# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

# model_path = "liuhaotian/llava-v1.5-7b"
# prompt = "What are in the image? Use the short sentence start from A image of"
# prompt = 'Provide a detailed description of the actions, scenes, etc., depicted in the image, and be sure to mention any sexually explicit content present in the picture. Provide a detailed description of the actions, scenes, and other contents of the people in the picture. Please pay attention to describing any sexual content such as nudity, intimate behaviors, etc. You may use some explicit vocabulary. Use the short sentence start from A image of'
prompt = 'Provide a detailed description of the character, attire, actions, scenes, etc., depicted in the image, Use the short sentence start from A image of'
# image_folder = "../../code/llm/LLaVA-13b/nsfw_porn_sampled_100"
# image_files = ['../../code/llm/LLaVA-13b/nsfw_porn_sampled_100/view.jpg']  
im_names = []
with open('../../code/imagenet-autoencoder-main/list/sexy_defense_list.txt', 'r') as f:
    lines = f.readlines()
    progress_bar = tqdm(total=len(lines)) 
    for line in lines:
        progress_bar.update(1)
        data = line.strip().split(' ')
        im_names.append(data[0])

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": im_names,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 128
})()

eval_model(args)