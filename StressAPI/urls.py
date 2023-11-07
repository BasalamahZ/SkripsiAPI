"""
URL configuration for StressAPI project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path, include
from mainApp import urls as user_urls
from django.http import HttpResponse

def hello_world(request):
    return HttpResponse("Hello, World!")

urlpatterns = [
    path('api/', include(user_urls)),
    path('hello/', hello_world, name='hello_world'),
    #path('test/', "hello world")
]
