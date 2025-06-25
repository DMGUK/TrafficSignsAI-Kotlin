package com.example.trafficsignapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import ai.onnxruntime.*
import android.text.SpannableString
import android.text.Spanned
import android.text.style.StyleSpan
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.roundToInt
import android.view.Surface // Import Surface for rotation constants

/**
 * Main activity for the Traffic Sign Detection application.
 * This version aligns camera frames with the display orientation and sets ImageAnalysis
 * resolution to match the ONNX model's input size. It passes the current display rotation
 * to the OverlayView for accurate drawing of bounding boxes, handling display orientation changes.
 * Pixel values are normalized to [0, 1] range using standard RGB channel order.
 * NNAPI remains disabled for stability.
 */
class MainActivityOnnx : AppCompatActivity() {

    // ONNX Runtime components
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var ortSession: OrtSession

    // Model and UI related properties
    private lateinit var labels: List<String> // List of labels for traffic signs
    private lateinit var inputName: String // Name of the input tensor for the ONNX model
    private var inputWidth: Int = 0 // Expected input width of the ONNX model (e.g., 736)
    private var inputHeight: Int = 0 // Expected input height of the ONNX model (e.g., 736)
    private var inputChannels: Int = 3 // Expected input channels (e.g., 3 for RGB)

    private lateinit var overlay: OverlayView // Custom view to draw bounding boxes
    private lateinit var cameraExecutor: java.util.concurrent.ExecutorService // Executor for camera operations
    private lateinit var tvLabel: TextView // TextView to display detected labels
    private lateinit var tvProb: TextView // TextView to display detection probabilities (currently unused in UI)
    private lateinit var previewView: PreviewView // Reference to the PreviewView

    // TARGET_ANALYSIS_WIDTH/HEIGHT will be dynamically set to match inputWidth/inputHeight
    private var TARGET_ANALYSIS_WIDTH: Int = 0
    private var TARGET_ANALYSIS_HEIGHT: Int = 0


    // Companion object for constants
    companion object {
        private const val TAG = "TrafficSignAI" // Tag for logging
        private const val REQUEST_CODE_PERMISSIONS = 10 // Request code for camera permissions
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA) // Required permissions
    }

    /**
     * Called when the activity is first created.
     * Initializes UI components, loads the ONNX model, sets up camera, and requests permissions.
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements by finding them in the layout
        tvLabel = findViewById(R.id.tvLabel)
        tvProb = findViewById(R.id.tvProbability)
        overlay = findViewById(R.id.overlay)
        previewView = findViewById(R.id.previewView) // Get reference to PreviewView

        // Initialize ONNX Runtime environment
        ortEnv = OrtEnvironment.getEnvironment()
        try {
            // Load the ONNX model from assets. 'best.onnx' should be in the 'src/main/assets' folder.
            val modelBytes = assets.open("best.onnx").use { it.readBytes() }

            // Configure ONNX Session Options - explicitly NOT adding NNAPI here.
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnv.createSession(modelBytes, sessionOptions)
            Log.d(TAG, "ONNX model 'best.onnx' loaded successfully with CPU execution provider (NNAPI disabled).")

            // Extract input tensor information from the ONNX model
            val firstInput = ortSession.inputInfo.values.first()
            inputName = firstInput.name // Get the name of the input layer
            val tinfo = firstInput.info as TensorInfo // Get tensor information
            inputChannels = tinfo.shape[1].toInt() // Get number of input channels (e.g., 3 for RGB)
            inputHeight = tinfo.shape[2].toInt() // Get input height
            inputWidth = tinfo.shape[3].toInt() // Get input width
            Log.d(TAG, "Model input info: Name=$inputName, Shape=[1, $inputChannels, $inputHeight, $inputWidth] (Expected elements: ${1L * inputChannels * inputHeight * inputWidth})")

            // Set TARGET_ANALYSIS_WIDTH/HEIGHT to match model input resolution directly.
            TARGET_ANALYSIS_WIDTH = inputWidth
            TARGET_ANALYSIS_HEIGHT = inputHeight
            Log.d(TAG, "TARGET_ANALYSIS_RESOLUTION set to model input size: ${TARGET_ANALYSIS_WIDTH}x${TARGET_ANALYSIS_HEIGHT} (matches model input)")


        } catch (e: Exception) {
            Log.e(TAG, "Error loading ONNX model or getting input info: ${e.message}", e)
            tvLabel.text = "Error: Model loading failed!"
            return // Stop execution if model fails to load
        }

        try {
            // Load labels from a text file. 'labels.txt' should be in the 'src/main/assets' folder.
            labels = assets.open("labels.txt")
                .bufferedReader()
                .useLines { it.toList() }
            Log.d(TAG, "Labels loaded successfully. Total labels: ${labels.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels.txt: ${e.message}", e)
            tvLabel.text = "Error: Labels loading failed!"
            return // Stop execution if labels fail to load
        }


        // Initialize a single-thread executor for camera operations to process frames sequentially
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Check for camera permissions. If granted, start the camera; otherwise, request them.
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
    }

    /**
     * Called when the activity is being destroyed.
     * Releases ONNX Runtime resources and shuts down the camera executor.
     */
    override fun onDestroy() {
        super.onDestroy()
        ortSession.close()
        ortEnv.close()
        cameraExecutor.shutdown()
        Log.d(TAG, "ONNX Runtime session and environment closed. Camera executor shut down.")
    }

    /**
     * Processes a single camera frame (ImageProxy), converts it to a Bitmap,
     * runs ONNX model inference, and updates the UI with detection results.
     *
     * @param imageProxy The ImageProxy representing the current camera frame.
     */
    private fun processCameraFrame(imageProxy: ImageProxy) {
        // Ensure ImageProxy is always closed, even if an exception occurs
        try {
            val inferenceStartTime = System.currentTimeMillis() // Start timing inference

            // Step 1: Preprocess the ImageProxy to get a FloatBuffer of the correct size
            val preparedInput = preprocessAndPrepareTensor(imageProxy)
            if (preparedInput == null) {
                Log.e(TAG, "Failed to prepare tensor input for ONNX model. Skipping frame.")
                return // Exit if input preparation fails
            }

            // Destructure the Triple to get the FloatBuffer and original ImageProxy dimensions
            val (inputBuffer, originalImageWidth, originalImageHeight) = preparedInput

            // Step 2: Create an ONNX tensor from the preprocessed float array
            val tensor = OnnxTensor.createTensor(
                ortEnv,
                inputBuffer,
                // Define the shape of the tensor, which MUST match the model's expected input shape
                longArrayOf(1, inputChannels.toLong(), inputHeight.toLong(), inputWidth.toLong())
            )

            // Step 3: Run inference with the ONNX session
            val ortResults = ortSession.run(mapOf(inputName to tensor))

            val inferenceEndTime = System.currentTimeMillis() // End timing inference
            val inferenceTimeMs = inferenceEndTime - inferenceStartTime
            Log.d(TAG, "ONNX Inference time (including pre-processing and tensor creation): $inferenceTimeMs ms")

            // Update UI with inference time
            runOnUiThread {
                val prefix = "Inference time: "
                val fullText = prefix + "${inferenceTimeMs} ms"
                val spannableString = SpannableString(fullText)

                spannableString.setSpan(
                    StyleSpan(Typeface.BOLD),
                    0,
                    prefix.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                tvProb.text = spannableString
            }

            // Step 4: Post-process the raw model output to extract meaningful detection boxes
            val rawDetections = postprocess(ortResults)

            // Step 5: Update UI elements on the main thread
            runOnUiThread {
                // MODIFICATION START: Display class names and probabilities in tvLabel
                if (rawDetections.isNotEmpty()) {
                    val detectionStrings = rawDetections.map { detection ->
                        "${detection.classId}. ${detection.label}: ${"%.0f".format(detection.confidence * 100)}%"
                    }
                    tvLabel.text = "Detections:\n" + detectionStrings.joinToString("\n")
                } else {
                    tvLabel.text = "No detections"
                }
                // MODIFICATION END

                // Pass the model's input dimensions AND the PreviewView's transformation matrix
                // to the overlay view. The OverlayView will use this matrix to correctly map
                // model coordinates to screen coordinates.
                overlay.setTransformationInfo(
                    inputWidth, // This is model's input width
                    inputHeight, // This is model's input height
                    previewView.width, // Current width of the PreviewView on screen
                    previewView.height, // Current height of the PreviewView on screen
                    previewView.matrix // The transformation matrix of PreviewView
                )
                overlay.setVerticalAdjustment(200f)
                overlay.setHorizontalAdjustment(150f)
                overlay.boxes = rawDetections // Provide detection boxes to the OverlayView
                overlay.invalidate() // Request a redraw of the overlay to show new boxes
            }

            // Step 6: Close resources
            tensor.close()
            ortResults.close()

        } catch (e: Exception) {
            // Catch any exceptions during the entire processing pipeline
            Log.e(TAG, "Exception during processCameraFrame() pipeline: ${e.message}", e)
        } finally {
            // This block will always execute, ensuring the ImageProxy is closed.
            imageProxy.close()
        }
    }


    /**
     * Converts a YUV_420_888 ImageProxy to a Bitmap.
     *
     * @param imageProxy The ImageProxy to convert.
     * @return A Bitmap representation of the image, or null if conversion fails.
     */
    private fun yuvToBitmap(imageProxy: ImageProxy): Bitmap? {
        // Check if the image format is YUV_420_888
        if (imageProxy.format != ImageFormat.YUV_420_888) {
            Log.e(TAG, "Unsupported image format: ${imageProxy.format}")
            return null
        }

        // Get buffers for Y, U, and V planes
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        // Get the size of each buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // Create a ByteArray to hold NV21 data (YUV_420_888 is converted to NV21 format)
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize) // Copy Y plane
        vBuffer.get(nv21, ySize, vSize) // Copy V plane
        uBuffer.get(nv21, ySize + vSize, uSize) // Copy U plane

        // Create a YuvImage from the NV21 data
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        // Compress the YuvImage to JPEG
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        // Decode the JPEG byte array into a Bitmap
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    /**
     * Helper function to rotate a Bitmap.
     * @param degrees The rotation angle.
     * @return The rotated Bitmap.
     */
    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    // ... (keep onCreate, onDestroy, processCameraFrame, yuvToBitmap)

    /**
     * Preprocesses the incoming ImageProxy, ROTATES it to be upright, scales it,
     * and converts it into a FloatBuffer suitable for ONNX Runtime.
     *
     * @param imageProxy The ImageProxy from the camera.
     * @return A Triple containing the FloatBuffer for the ONNX model, and the original
     * width and height of the ImageProxy, or null if processing fails.
     */
    private fun preprocessAndPrepareTensor(imageProxy: ImageProxy): Triple<FloatBuffer, Int, Int>? {
        // Get the rotation degrees from the ImageProxy's metadata. This is the
        // reliable way to determine the orientation of the image buffer.
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        val originalBitmap = yuvToBitmap(imageProxy)
        originalBitmap ?: return null // Return null if bitmap conversion fails

        // Rotate the bitmap if necessary to ensure it's upright.
        val rotatedBitmap = if (rotationDegrees != 0) {
            originalBitmap.rotate(rotationDegrees.toFloat())
        } else {
            originalBitmap
        }

        // Now, the `rotatedBitmap` is correctly oriented. Scale it to the model's input size.
        val processedBitmapForModel = Bitmap.createScaledBitmap(rotatedBitmap, inputWidth, inputHeight, true)
        Log.d(TAG, "Bitmap created/scaled for model input in preprocessAndPrepareTensor: ${processedBitmapForModel.width}x${processedBitmapForModel.height} (Expected: ${inputWidth}x${inputHeight})")

        val numPixels = inputWidth * inputHeight
        val px = IntArray(numPixels)

        processedBitmapForModel.getPixels(px, 0, inputWidth, 0, 0, inputWidth, inputHeight)

        val out = FloatArray(numPixels * inputChannels) // Initialize FloatArray with exact expected size

        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixelIndexInPx = y * inputWidth + x // Index in the 1D pixel array (ARGB)
                val color = px[pixelIndexInPx] // Get the ARGB integer for the current pixel

                // Extract R, G, B components
                val r = (color shr 16) and 0xFF
                val g = (color shr 8) and 0xFF
                val b = color and 0xFF

                val floatArrayIndex = y * inputWidth + x // Linear index for each color plane

                // Apply [0, 1] normalization with RGB channel order (common for many models)
                out[floatArrayIndex] = r / 255.0f // Red channel to the first plane
                out[floatArrayIndex + numPixels] = g / 255.0f // Green channel to the second plane
                out[floatArrayIndex + (2 * numPixels)] = b / 255.0f // Blue channel to the third plane
            }
        }

        // IMPORTANT: Recycle all created bitmaps to avoid memory leaks
        processedBitmapForModel.recycle()
        if (rotatedBitmap !== originalBitmap) {
            rotatedBitmap.recycle()
        }
        originalBitmap.recycle()

        Log.d(TAG, "FloatBuffer size generated: ${out.size} elements (Expected: ${1L * inputChannels * inputHeight * inputWidth})")

        return Triple(FloatBuffer.wrap(out), imageProxy.width, imageProxy.height)
    }

    /**
     * Post-processes the raw output from the ONNX model to extract a list of detection boxes.
     * It filters detections based on a confidence threshold.
     * The model output is expected to be in the format: [x1, y1, x2, y2, confidence, class_id].
     *
     * @param results The OrtSession.Result object containing the model's output tensor.
     * @param confThreshold The minimum confidence score (0.0 to 1.0) for a detection to be included.
     * @return A list of `DetectionBox` objects representing the valid detections.
     */
    private fun postprocess(
        results: OrtSession.Result,
        confThreshold: Float = 0.25f // Increased threshold for cleaner detections
    ): List<DetectionBox> {
        val detections = mutableListOf<DetectionBox>()
        try {
            // The model output is assumed to be a 3D array: [batch_size, num_detections, 6]
            // where 6 represents: x1, y1, x2, y2, confidence, class_id
            @Suppress("UNCHECKED_CAST")
            val output = results[0].value as Array<Array<FloatArray>>

            // Log the first few raw output elements for debugging
            Log.d(TAG, "--- Raw Model Output (First 5 Detections) ---")
            for (i in 0 until min(5, output[0].size)) {
                val box = output[0][i]
                val confidence = box[4] // Assuming confidence is at index 4
                Log.d(TAG, "Output $i: Confidence=${"%.4f".format(confidence)}, Box=[${"%.2f".format(box[0])}, ${"%.2f".format(box[1])}, ${"%.2f".format(box[2])}, ${"%.2f".format(box[3])}]")
            }


            val detectedBoxes = output[0] // Assuming batch_size is 1, get detections for the first image

            Log.d(TAG, "--- Raw Detections from Model (Before Threshold) ---")
            for (i in detectedBoxes.indices) {
                val box = detectedBoxes[i]
                val confidence = box[4] // Extract the confidence score from the box array
                val classId = box[5].toInt() // Extract the class ID
                val label = if (classId in labels.indices) labels[classId] else "Unknown"

                // Log: Output class name and probability (as requested for detections)
                Log.d(TAG, "Raw Detection $i: Label=$label, Confidence=${"%.2f".format(confidence * 100)}%, Rect=[${"%.2f".format(box[0])}, ${"%.2f".format(box[1])}, ${"%.2f".format(box[2])}, ${"%.2f".format(box[3])}]")

                if (confidence >= confThreshold) {
                    // Extract bounding box coordinates (left, top, right, bottom)
                    val left = box[0]
                    val top = box[1]
                    val right = box[2]
                    val bottom = box[3]

                    // Ensure the class ID is valid and corresponds to an existing label
                    if (classId in labels.indices) {
                        detections.add(
                            DetectionBox(
                                rect = RectF(left, top, right, bottom),
                                classId = classId,
                                label = labels[classId], // Pass the actual label for display
                                confidence = confidence
                            )
                            // Note: These coordinates are relative to the model's inputWidth/inputHeight
                        )
                    }
                }
            }
            Log.d(TAG, "--- Filtered Detections (After Threshold, Count: ${detections.size}) ---")

        } catch (e: Exception) {
            // Log any errors during post-processing
            Log.e(TAG, "Error during postprocessing: ${e.message}", e)
        }
        return detections
    }

    /**
     * Checks if all required camera permissions (defined in `REQUIRED_PERMISSIONS`) are granted.
     *
     * @return `true` if all permissions are granted, `false` otherwise.
     */
    private fun allPermissionsGranted() =
        REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

    /**
     * Configures and starts the camera preview and image analysis use cases.
     * It binds the camera to the activity's lifecycle.
     */
    private fun startCamera() {
        // Get the singleton instance of ProcessCameraProvider
        val camProviderF = ProcessCameraProvider.getInstance(this)
        camProviderF.addListener({
            val camProvider = camProviderF.get() // Get the camera provider instance

            // Build the Preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    // Set the surface provider for the preview to display camera feed
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // The display rotation from the PreviewView, used by ImageAnalysis for correct orientation.
            val displayRotation = previewView.display.rotation

            // Build the ImageAnalysis use case
            val analysis = ImageAnalysis.Builder()
                // Set the target resolution for image analysis frames to match model input.
                // This ensures the camera provides frames at the exact size the model expects,
                // minimizing quality degradation from unnecessary scaling.
                .setTargetResolution(Size(TARGET_ANALYSIS_WIDTH, TARGET_ANALYSIS_HEIGHT))
                // Set the target rotation for the analyzer to match the display orientation.
                // This ensures the bitmap received by the analyzer is already rotated correctly.
                .setTargetRotation(displayRotation)
                // Set backpressure strategy to keep only the latest frame, dropping older ones
                // This is crucial for real-time performance and avoiding processing old frames.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    // Set the analyzer for image analysis.
                    it.setAnalyzer(cameraExecutor, ImageAnalyzer(this@MainActivityOnnx))
                }

            // Select the back camera by default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Unbind any previous use cases to prevent conflicts
            camProvider.unbindAll()
            try {
                // Bind the camera selector, preview, and analysis use cases to the activity's lifecycle
                camProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    analysis
                )
                // No need to pass rotation to overlay directly; it will get the matrix from previewView
            } catch (exc: Exception) {
                // Log any errors that occur during camera binding
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this)) // Ensure the listener runs on the main thread
    }

    /**
     * Custom `ImageAnalysis.Analyzer` implementation that simply forwards the `ImageProxy`
     * to the `MainActivityOnnx` instance for processing.
     */
    private class ImageAnalyzer(private val mainActivity: MainActivityOnnx) : ImageAnalysis.Analyzer {
        /**
         * Analyzes an image frame from the camera.
         * This method is called by CameraX for each new camera frame.
         * It forwards the `ImageProxy` to the `mainActivity` for full processing.
         *
         * @param imageProxy The `ImageProxy` representing the current camera frame.
         */
        override fun analyze(imageProxy: ImageProxy) {
            mainActivity.processCameraFrame(imageProxy)
        }
    }
}