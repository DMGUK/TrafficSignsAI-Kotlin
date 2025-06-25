TrafficSignsAI-Kotlin
This repository contains the Android application for real-time traffic sign detection using a pre-trained ONNX model. It serves as the deployment pipeline for models trained externally (e.g., with YOLOv8 via Python, as described in the companion TrafficSignsAI-Python project), demonstrating on-device inference using ONNX Runtime Mobile and CameraX.

📁 Project Structure
TrafficSignsAI-Kotlin/
│
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── AndroidManifest.xml       # App permissions and configuration
│   │   │   ├── java/
│   │   │   │   └── com/example/trafficsignapp/
│   │   │   │       ├── MainActivityOnnx.kt # Main activity handling camera, inference, and UI updates
│   │   │   │       └── OverlayView.kt      # Custom view for drawing detection bounding boxes and labels
│   │   │   └── assets/                     # Pre-trained ONNX model and class labels
│   │   │       ├── best.onnx               # The ONNX inference model
│   │   │       └── labels.txt              # Text file containing class labels (one per line)
│   │   └── res/                            # Android resources (layouts, drawables, etc.)
│   │       └── layout/
│   │           └── activity_main.xml       # Main layout for the camera preview and overlay
│   ├── build.gradle                        # Module-level Gradle build file (dependencies, Android configurations)
│   └── ...                                 # Other standard Android project files
├── build.gradle                            # Project-level Gradle build file
└── README.md                               # This file

🚀 Features
Real-time Object Detection: Performs live traffic sign detection directly on Android devices.

CameraX Integration: Utilizes the CameraX library for efficient and robust camera preview and image analysis.

ONNX Runtime Mobile: Leverages ONNX Runtime Mobile for high-performance and optimized ONNX model inference on Android.

Custom Overlay Drawing: Implements a custom OverlayView to precisely draw bounding boxes and confidence labels on top of the camera feed.

Dynamic Alignment Adjustments: Provides configurable horizontal and vertical offsets within the OverlayView to fine-tune the alignment of detections with the camera preview.

Optimized Pre-processing: Includes robust image pre-processing (YUV to Bitmap conversion, rotation, scaling, normalization) to prepare camera frames for model input.

Efficient Post-processing: Extracts and filters detection results from the ONNX model's output tensor, applying confidence thresholds.

🧩 Requirements
Android Studio: Latest version recommended for Android development.

Kotlin: The programming language used for the application.

Android SDK: Target API Level 21+ (minSdkVersion) and compile against latest stable SDK (e.g., API 34).

Gradle: The build automation tool for Android projects.

ONNX Runtime Mobile AAR: Dependencies for ONNX inference are managed via Gradle (app/build.gradle).

Pre-trained ONNX Model: A best.onnx file (e.g., from YOLOv8 training) and a labels.txt file compatible with your model's output, placed in app/src/main/assets/.