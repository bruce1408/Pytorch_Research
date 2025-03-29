import os
import onnx
import torch
import argparse
from tqdm import tqdm
import onnxruntime
import numpy as np
import PIL,time
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import common.config as config_pytorch
from spectrautils.print_utils import print_colored_box
os.environ["CUDA_VISIBLE_DEVICES"]="3,4"

# 1. load config variable and load model from pytorch
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch convert onnx to verficaiton Acc')
    
    # 模型选择
    parser.add_argument('--model_name', default="resnet18", help='model name resnet-50、mobilenet_v2、efficientnet')
    parser.add_argument('--torch_path', default="/root/resnet18-f37072fd.pth", help='laod checkpoints from saved models')
    parser.add_argument('--onnx_path', default="/share/cdd/onnx_models/resnet18_official.onnx", help="pth model convert to onnx name")
    parser.add_argument('--input_shape', type=list, nargs='+', default=[1, 3, 224, 224])
    parser.add_argument("--label_path", default="/root/val_torch_cls/synset.txt", help="label path")
    parser.add_argument('--data_val', default="/mnt/share_disk/bruce_trie/workspace/outputs/imagenet_dataset/val", help="val data")


    args = parser.parse_args()

    return args


# img preprocessing method
def pre_process_img():
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    
    return transform


# get the val data and preprocessing 
def onnxruntime_infer():
    print_colored_box("loading the data and preprocessing the image data ...")
    
    count = 0 
    total_time = 0
    correct_count = 0
    dir_name_list = sorted(os.listdir(args.data_val))
    
    providers = ['CUDAExecutionProvider']

    sess = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    
    transform = pre_process_img()
    
    for true_label, dir_ in enumerate(dir_name_list):
        img_dir = os.path.join(args.data_val, dir_)
        for img_name in os.listdir(img_dir):
            count += 1
            img_path =os.path.join(img_dir, img_name)
            img = PIL.Image.open(img_path).convert("RGB")
            img_data = transform(img).unsqueeze(0).numpy()
            output = sess.run(None, {input_name: img_data})[0]
            predicted_class = np.argmax(output)
            if predicted_class == true_label:
                correct_count += 1
        
        if count % 1000 == 0 :
            print("the acc is {}".format(correct_count / count))
            
    accuracy = correct_count / count
    avg_time_per_image = total_time / count
    print(f'Classification accuracy: {accuracy:.4f}')
    print(f'Average inference time per image: {avg_time_per_image*1000:.2f} ms')
    

    
# get the torch official models 
def get_torch_model():
    # model = models.__dict__[args.model_name](pretrained=True)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # state_dict = torch.load(args.model_file)
    # model.load_state_dict(state_dict)
    return model


# export .pth model to .onnx
def export_onnx(model, input_shape, export_onnx_path):
    model = get_torch_model()
    model.eval()
    torch.onnx.export(
        model, 
        torch.randn(input_shape),
        export_onnx_path, 
        input_names=["input"], 
        output_names=["output"], 
        opset_version=11
    )
    print_colored_box("Onnx model has been exported success!")


# load onnx and val the data with ort
def load_onnx_and_eval(test_images, img_labels):
    sess = onnxruntime.InferenceSession(args.export_path)
    correct_count = 0
    total_count = len(test_images)
    print("begin to eval the model...")
    for i in tqdm(range(total_count)):
        input_data = np.expand_dims(test_images[i], axis=0).astype(np.float32)
        output = sess.run(None, {'img': input_data})[0]
        predicted_class = np.argmax(output)
        # predict = img_dict[predicted_class]
    
        if predicted_class == img_labels[i]:
            correct_count += 1
            
    accuracy = correct_count / total_count
    print_colored_box('Classification accuracy:', accuracy)


if __name__ == "__main__":
    args = parse_args()
    
    #  onnxruntime 推理
    onnxruntime_infer()
    
    #  加载torch模型
    model = get_torch_model()
    
    #  导出onnx模型
    export_onnx(model, args.input_shape, args.onnx_path)
    