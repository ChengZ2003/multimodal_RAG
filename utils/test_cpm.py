from img_convertor.cpm_convertor import CPMConvertor

model_path = "/home/project/data/jc/mmRAG/model/MiniCPM-Llama3-V-2_5"

convertor  = CPMConvertor(model_path, device='cuda:1')

img_to_process = '/home/project/data/jc/mmRAG/MiniCPM_Llama3/figure-1-5.jpg'
query = 'Please convert the image to a markdown syntax table'
res = convertor.convert(img_path = img_to_process,
                        query = query)
print(res)