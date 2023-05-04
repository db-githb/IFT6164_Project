import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torchvision.io import read_image
from resnet import ResNet34

os.chdir("C:/Users/Damian/Documents/School/Winter 2023 - Grad School/IFT 6164 - Adversarial Learning/Project/StyleGAN_pytorch")
def load_data(dp_prefix):
    img_lst = []
    for c in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]: # "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        datapath = dp_prefix+c
        folder_name_lst = os.listdir(datapath)
        for fldr_name in folder_name_lst:
            img_lst.append(datapath+"/"+fldr_name+"/target.png")
    return img_lst

img_lst = load_data("out_fgsm/")
model = torch.load('resnet34_original_cifar_8_more_data_aug.pt')
model.eval()
count = 0
print(count)
f = open("predicted_classes.json", "w")
f.write("{\n")
for i in img_lst:
    img = read_image(i).unsqueeze(0)
    img_tensor = img.type(torch.FloatTensor).to("cuda:0")/255
    with torch.no_grad():
        pred_vec = model(img_tensor)
        max_pred = torch.argmax(pred_vec, dim=1)
        pred = max_pred.to("cpu").numpy()[0]
        f.write("\""+str(count)+"\":"+str(pred)+",\n")
        count +=1
        if count % 50 == 0:
            count+=200
            print(count)
f.write("}")
f.close()