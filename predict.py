import functions
import argparse

parser = argparse.ArgumentParser(description='parser_for_predict')
parser.add_argument('image_path',  default="./flowers/test/100/image_07896.jpg", type=str)
parser.add_argument('checkpoint', default="./checkpoint.pth", type=str)
parser.add_argument('--top_k', dest="top_k", default = 5, type=int)
parser.add_argument('--category_names', dest="category_names", default='cat_to_name.json')
parser.add_argument('--gpu', dest="gpu", default=True, type=bool)
parsed = parser.parse_args()
image_path = parsed.image_path
checkpoint = parsed.checkpoint
top_k = parsed.top_k
category_names = parsed.category_names
gpu = parsed.gpu

model, optimizer, idx_to_class = functions.load_checkpoint(checkpoint)
cat_to_name = functions.map_label(category_names)
functions.predict(image_path, model, top_k, gpu, cat_to_name, idx_to_class)