#导入StableDiffusionPipeline相关的包
from diffusers import StableDiffusionPipeline   
import torch

#加载stable diffusion模型

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",torch_dtype=torch.float16)

pipe = pipe.to('cuda') #将模型放到cuda上  需要电脑gpu支持 

# 生成图片
prompt = "a photo of a dog"
image = pipe(prompt).images[0]

# 保存图片
image.save("dog.jpg")
print("图片保存成功")