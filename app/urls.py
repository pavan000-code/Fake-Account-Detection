from django.conf.urls import url
from django.urls import path
from .import views

urlpatterns=[
    path('',views.index,name='index'),
    path('home',views.home,name='home'),
    path('model',views.model,name='model'),
    path('prediction',views.prediction,name='prediction'),
    path('graphe',views.graphs,name='graphe')







]