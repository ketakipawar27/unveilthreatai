from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, serializers
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .serializers import UserProfileSerializer
from rest_framework import status

from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework.permissions import AllowAny
from .serializers import UserSignupSerializer







User = get_user_model()


# ----------------- Signup -----------------
@api_view(["POST"])
@permission_classes([AllowAny])  # ✅ allow anyone to access signup
@csrf_exempt
def signup(request):
    serializer = UserSignupSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# ----------------- Login -----------------
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    # Use default username_field = 'username'
    def validate(self, attrs):
        username = attrs.get("username")
        password = attrs.get("password")

        if not username or not password:
            raise serializers.ValidationError("Username and password are required")

        # Authenticate user
        from django.contrib.auth import authenticate
        user = authenticate(username=username, password=password)
        if not user:
            raise serializers.ValidationError("Invalid username or password")

        # Map username for JWT
        attrs['username'] = user.username
        return super().validate(attrs)


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        serializer = UserProfileSerializer(user, context={"request": request})
        return Response(serializer.data)

    def put(self, request):
        user = request.user
        data = request.data.copy()
        if 'avatar' in request.FILES:
            data['avatar'] = request.FILES['avatar']

        serializer = UserProfileSerializer(user, data=data, partial=True, context={"request": request})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
