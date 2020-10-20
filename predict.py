import argparse
import json
import numpy as np
import torch

from model_utils import load_checkpoint 
from data_utils import process_image

# parse args from command line
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Parser command line arguments for Flower Image Classifier',
    )
    parser.add_argument('image_path', type=str , help='image location')
    parser.add_argument('checkpoint', type=str , help='saved model checkpoint')
    parser.add_argument('--top_k', type=int , default=5, help='Top k most probable classes')
    parser.add_argument('--category_names', type=str , default='cat_to_name.json', help='Map of labels to names')
    parser.add_argument('--gpu', type=bool , default=False, help='use a GPU')

    return parser.parse_args()


def main():
    # label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # load model checkpoint
    model = load_checkpoint(args.checkpoint)
    print("Loaded model: \n{}".format(model))

    # predict flower names
    probs, classes = predict(args.image_path, args.checkpoint, args.top_k)
    names = [cat_to_name[str(idx)] for idx in classes]
    
    # print
    print('Top {} most probably classes!'.format(args.top_k))
    print('Names: {}'.format(names))
    print('Probabilities: {}'.format(probs))
    

# predict
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if args.gpu else 'cpu')

    # process image
    processed_image = process_image(image_path).unsqueeze_(0)
    processed_image = processed_image.to(device)

    # load model from checkpoint
    model = load_checkpoint(model)
    model.to(device)
    model.eval()
        
    with torch.no_grad():
        output = model.forward(processed_image)
        ps = torch.exp(output)
    
        top_probs, top_idx = ps.topk(topk, dim=1)
    
        top_probs_array = np.array(top_probs)[0]
        top_idx_array = np.array(top_idx)[0]
        
    model.train()
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        
    top_classes = [idx_to_class[i] for i in top_idx_array]
    
    return top_probs_array, top_classes


# Example command: python predict.py flowers/test/13/image_05745.jpg saved_models/checkpoint.pth --gpu true --top_k 5
if __name__ == "__main__":
    args = parse_arguments()
    main()
