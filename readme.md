# EastAsiaTrafficSignsAI-Kotlin

This repository contains the Android mobile application for real-time road sign detection using an ONNX-based YOLOv8 model. It complements the Python-based dataset generation and training pipeline from [TrafficSignsAI-Python](https://github.com/DMGUK/TrafficSignsAI-Python).

## Features

- Real-time camera preview and detection using ONNX Runtime
- Lightweight architecture suitable for mobile devices
- Displays top detected road sign classes and confidence scores
- Easy integration with custom YOLOv8 models exported to ONNX

## Requirements

- Android Studio (Arctic Fox or later)
- Android SDK 21+
- Kotlin
- CameraX
- ONNX Runtime for Mobile

## Folder Structure

```
TrafficSignsAI-Kotlin/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/trafficsignapp/
│   │   │   │   ├── MainActivityOnnx.kt
│   │   │   │   └── OverlayView.kt
│   │   │   └── res/
│   │   │       ├── layout/activity_main.xml
│   │   │       └── values/strings.xml
│   └── build.gradle
├── build.gradle
└── README.md
```

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/DMGUK/TrafficSignsAI-Kotlin
   cd TrafficSignsAI-Kotlin
   ```

2. Open the project in Android Studio.

3. Download your quantized YOLOv8 ONNX model (`best_int8.onnx`) and place it in:
   ```
   app/src/main/assets/
   ```

4. Ensure required permissions are set in `AndroidManifest.xml` (e.g., camera access).

5. Build and run the app on a physical device (CameraX requires a real camera input).

## Detection Output Format

The app expects ONNX model outputs in the format:
```
[x_center, y_center, width, height, object_conf, class_id]
```
(Shape: [1, 300, 6] for YOLOv8)

## Related Repositories

- [TrafficSignsAI-Python](https://github.com/DMGUK/TrafficSignsAI-Python) - Dataset creation, training and export to ONNX

## License

This project is licensed under the MIT License.
