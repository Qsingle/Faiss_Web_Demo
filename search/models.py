from django.db import models
from pymongo import MongoClient
import json
import torch
from torchvision import transforms
from PIL import Image
import os
from .modules import resnet
# Create your models here.
class Upload(models.Model):
    name = models.CharField(max_length=50)
    img = models.FileField(upload_to="images")


class Search:
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(__file__))
        configfile = os.path.join(self.dir,"config.json")
        with open(configfile) as f:
            self.config =json.load(f)
        host = self.config["db"]["host"]
        port = self.config["db"]["port"]
        self.client = MongoClient(host=host,port=port)
        self.db = self.client[self.config["db"]["database"]]
        self.col = self.db.get_collection(self.config["db"]["collection"])

    def search(self,x):
        import faiss
        index = faiss.read_index(os.path.join(self.dir,self.config["index"]))
        model = resnet.ResNet(self.config["channel"],n_layers=self.config["layers"])
        print("load model successful")
        model.fc = torch.nn.ReLU(inplace=False)
        model_path = os.path.join(self.dir,self.config["model"])
        if self.config["use_gpu"]:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        state = torch.load(model_path,map_location=device)
        load_state = {k : v for k,v in state.items() if 'fc' not in k}
        model_state = model.state_dict()
        model_state.update(load_state)
        model.load_state_dict(model_state)
        #print("load weights successful")
        model.to(device)
        #print("model to device successful")
        #print(os.path.exists(x))
        img = Image.open(x)
        img = img.convert("RGB")
        size = self.config["size"]
        #print(size)
        transform = transforms.Compose([
            transforms.Resize([size,size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        #print(img.shape[0])
        #print("transform successful")
        img = img.unsqueeze(0)
        with torch.no_grad():
            img.to(device)
            #print("img to device successful")
            out = model(img)
            #print("predict successful")
            del img,model
            out = out.cpu().detach().numpy()
            #print("future extracted")
            if self.config["use_gpu"]:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res,0,index)
            k = self.config["k"]
            D, ids = index.search(out,k)
        paths = []
        ids = ids[0]
        for id in ids:
            if id != -1:
                result = self.col.find_one({"id":int(id)})
                filename = result["filename"]
                if result["dirname"]:
                    path = os.path.join(self.config['map_dir'],result["dirname"],filename)
                    #path = os.path.join(self.config['data_dir'],result["dirname"],filename)
                else:
                    path = os.path.join(self.config['map_dir'],filename)
                paths.append(path)
        #print("search successful")
        return paths
