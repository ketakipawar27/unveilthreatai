from django.urls import path
from .views import signup, MyTokenObtainPairView, UserProfileView, analyze

urlpatterns = [
    path("signup/", signup, name="signup"),
    path("login/", MyTokenObtainPairView.as_view(), name="login"),  # JWT login
    path("user/profile/", UserProfileView.as_view(), name="user-profile"),

    path("analyze", analyze, name="analyze"),

]
