#-*- coding:utf-8 -*-
#!/usr/bin/env python

from django.urls import path
from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path('',views.index,name="index"),
    #path('search/',views.result,name="result")
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    urlpatterns + static(settings.MEDIA_URL,document=settings.MEDIA_ROOT)
