from django.urls import path
from . import views
from django.contrib.auth.views import (LoginView,
                                       LogoutView)

urlpatterns = [
    path('',views.CustomizeLoginView.as_view(), name="login"),
    path('register/',views.register,name='register'),
    path('home/',views.home,name='home'),
    path('logout/',LogoutView.as_view(next_page= 'login'), name="logout"),
    
    
]