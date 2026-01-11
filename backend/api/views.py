from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status, serializers
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated, AllowAny
from .serializers import UserProfileSerializer, UserSignupSerializer

from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

# 🔥 consequence graph builder (pure python, no AI)
from consequence_graph import build_consequence_graph

import os
import time
from django.conf import settings
from django.core.files.storage import default_storage


User = get_user_model()


# ----------------- Signup -----------------
@api_view(["POST"])
@permission_classes([AllowAny])
@csrf_exempt
def signup(request):
    serializer = UserSignupSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(
            {"message": "User created successfully"},
            status=status.HTTP_201_CREATED
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ----------------- Login -----------------
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        username = attrs.get("username")
        password = attrs.get("password")

        if not username or not password:
            raise serializers.ValidationError("Username and password are required")

        from django.contrib.auth import authenticate
        user = authenticate(username=username, password=password)
        if not user:
            raise serializers.ValidationError("Invalid username or password")

        attrs["username"] = user.username
        return super().validate(attrs)


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


# ----------------- User Profile -----------------
class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        serializer = UserProfileSerializer(
            request.user,
            context={"request": request}
        )
        return Response(serializer.data)

    def put(self, request):
        data = request.data.copy()
        if "avatar" in request.FILES:
            data["avatar"] = request.FILES["avatar"]

        serializer = UserProfileSerializer(
            request.user,
            data=data,
            partial=True,
            context={"request": request}
        )
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ==================================================
# AI ANALYSIS ENDPOINT (PUBLIC)
# ==================================================
@api_view(["POST"])
@permission_classes([AllowAny])
@csrf_exempt
def analyze(request):
    uploaded_file = request.FILES.get("file")
    caption = request.POST.get("caption", "")

    if not uploaded_file:
        return Response(
            {"error": "file is required"},
            status=status.HTTP_400_BAD_REQUEST
        )

    # ---------------- SAVE FILE ----------------
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    relative_path = os.path.join("uploads", uploaded_file.name)
    absolute_path = os.path.join(settings.MEDIA_ROOT, relative_path)

    with default_storage.open(relative_path, "wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    start_time = time.time()

    # ---------------- PIPELINE SELECTION ----------------
    file_ext = uploaded_file.name.lower().split(".")[-1]
    image_exts = ["jpg", "jpeg", "png", "bmp", "webp"]

    try:
        if file_ext in image_exts:
            import image_pipeline  # lazy import
            ai_output = image_pipeline.process_image(
                absolute_path,
                caption=caption
            )
            input_type = "image"
        else:
            import document_pipeline  # lazy import
            ai_output = document_pipeline.process_document(absolute_path)
            input_type = "document"

    except Exception as e:
        return Response(
            {"status": "error", "message": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    processing_time = int((time.time() - start_time) * 1000)

    # ==================================================
    # 🔥 BUILD CONSEQUENCE GRAPH (NEW)
    # ==================================================
    consequence_graph = build_consequence_graph(
        ai_output,
        input_type=input_type
    )

    # ---------------- RESPONSE ----------------
    return Response(
        {
            "status": "success",
            "input_type": input_type,
            "ai_output": ai_output,                 # ✅ unchanged
            "consequence_graph": consequence_graph, # 🔥 NEW
            "meta": {
                "processing_time_ms": processing_time
            }
        },
        status=status.HTTP_200_OK
    )
