from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
import os

from .models import Upload,Search
# Create your views here.

def index(request):
    if request.method == "POST":
        file = request.FILES.get('pic',None)
        if file:
            name = file.name
            img = Upload(name=name,img=file)
            if not os.path.exists(os.path.join(settings.MEDIA_ROOT,"images",name)):
                img.save()
            search = Search()
            images = search.search(os.path.join(settings.MEDIA_ROOT,"images",name))
            content = {
                "images" : images
            }
            return render(request,"result.html", content)
            #return HttpResponse("successful")
    return render(request,"index.html")

def result(request):
    if request.method == "POST":
        if "submit" in request:
            file = request.FILES.get('pic',None)
            if file:
                name = file.name
                img = Upload(name=name,img=file)
                if not os.path.exists(os.path.join(settings.MEDIA_ROOT,"images",name)):
                    img.save()
                search = Search()
            images = search.search(os.path.join(settings.MEDIA_ROOT,"images",name))
            content = {
                "images" : images
            }
            return render(request,"result.html", content)
    return render(request,"result.html",)
