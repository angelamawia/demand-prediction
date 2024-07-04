from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.register, name='signup'),
    path('login/', views.login_user, name='login'),
    path('main/', views.main_view, name='main'),
    path('predict_demand/', views.predict_demand, name='predict_demand'),
    path('prediction/<str:predicted_demand>/<input_data>/', views.show_prediction, name='show_prediction'),
    path('logout/', views.user_logout, name='logout'),


]
