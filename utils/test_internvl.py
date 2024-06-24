from img_convertor.internvl_convertor import InternVLConvertor

path = "/home/project/data/jc/mmRAG/model/Mini-InternVL-Chat-4B-V1-5"
convertor = InternVLConvertor(path, device = 'cuda:0')

img = '/home/project/data/jc/mmRAG/MiniCPM_Llama3/figure-1-5.jpg'
query = 'Please convert the image to a markdown syntax table'

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

res = convertor.convert(img, query, generation_config)

print(res)