package com.example.trafficsignapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.View
import kotlin.math.min
import kotlin.math.max
import kotlin.math.abs

/**
 * Data class representing a detected object (e.g., a traffic sign).
 *
 * @param rect The bounding box rectangle of the detected object in the model's input coordinate system.
 * @param classId The integer ID of the detected class.
 * @param label The string label of the detected class.
 * @param confidence The confidence score of the detection (0.0 to 1.0).
 */
data class DetectionBox(
    val rect: RectF,
    val classId: Int,
    val label: String,
    val confidence: Float
)

/**
 * Custom View responsible for drawing bounding boxes and labels on top of a camera preview.
 * It uses the transformation matrix from the PreviewView to correctly map bounding box coordinates
 * from the model's input space to the screen space, ensuring accurate alignment.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    // A list to hold the bounding boxes to be drawn. This list is updated by MainActivityOnnx.
    var boxes: List<DetectionBox> = listOf()
        set(value) {
            field = value
            // Request a redraw whenever the boxes list is updated.
            invalidate()
        }

    // These hold the dimensions of the model's input image (e.g., 736x736),
    // which the bounding box coordinates are relative to.
    private var modelInputWidth: Int = 1
    private var modelInputHeight: Int = 1

    // The dimensions of the PreviewView itself.
    private var previewViewWidth: Int = 1
    private var previewViewHeight: Int = 1

    // The transformation matrix from PreviewView. This maps original image coordinates
    // to the scaled and rotated screen coordinates of the PreviewView.
    private var previewTransformMatrix: Matrix = Matrix()

    // Adjustable vertical offset for the bounding box and label position.
    // Positive values move the box/label DOWN, negative values move them UP.
    private var verticalAdjustmentPx: Float = 0f

    // Adjustable horizontal offset for the bounding box and label position.
    // Positive values move the box/label RIGHT, negative values move them LEFT.
    private var horizontalAdjustmentPx: Float = 0f


    // Paint for drawing the bounding box rectangles.
    private val boxPaint = Paint().apply {
        color = Color.GREEN // Green color for the box
        style = Paint.Style.STROKE // Only draw the outline
        strokeWidth = 8f // Thickness of the box line
    }

    // Paint for the background of the label text.
    private val textBackgroundPaint = Paint().apply {
        color = Color.GREEN // Green color for the background
        style = Paint.Style.FILL // Fill the rectangle
    }

    // Paint for the label text itself.
    private val textPaint = Paint().apply {
        color = Color.BLACK // Black color for the text
        textSize = 40f // Font size
        style = Paint.Style.FILL // Fill the text
        typeface = Typeface.DEFAULT_BOLD // Bold font
    }

    /**
     * Sets the necessary transformation information from the camera preview.
     * This is the most critical function for accurate overlay drawing.
     *
     * @param inputW The width of the model's input (e.g., 736).
     * @param inputH The height of the model's input (e.g., 736).
     * @param pvWidth The current width of the PreviewView on screen.
     * @param pvHeight The current height of the PreviewView on screen.
     * @param matrix The transformation matrix from PreviewView.getMatrix().
     */
    fun setTransformationInfo(inputW: Int, inputH: Int, pvWidth: Int, pvHeight: Int, matrix: Matrix) {
        modelInputWidth = inputW
        modelInputHeight = inputH
        previewViewWidth = pvWidth
        previewViewHeight = pvHeight
        previewTransformMatrix.set(matrix) // Use set() to copy the matrix, not assign reference
        Log.d("OverlayView", "setTransformationInfo: modelInput=${modelInputWidth}x${modelInputHeight}, previewView=${previewViewWidth}x${previewViewHeight}, matrix=${previewTransformMatrix}")
        invalidate()
    }

    /**
     * Sets a vertical adjustment for the drawn bounding boxes and labels.
     * This is useful to fine-tune the vertical alignment if the default transformation
     * places elements too high or too low.
     *
     * @param offsetPx The offset in pixels. Positive values move elements down, negative values move them up.
     */
    fun setVerticalAdjustment(offsetPx: Float) {
        verticalAdjustmentPx = offsetPx
        invalidate() // Request a redraw with the new offset
    }

    /**
     * Sets a horizontal adjustment for the drawn bounding boxes and labels.
     * This is useful to fine-tune the horizontal alignment if the default transformation
     * places elements too far left or right.
     *
     * @param offsetPx The offset in pixels. Positive values move elements right, negative values move them left.
     */
    fun setHorizontalAdjustment(offsetPx: Float) {
        horizontalAdjustmentPx = offsetPx
        invalidate() // Request a redraw with the new offset
    }

    /**
     * Called when the view needs to draw its content.
     * This method maps the detected boxes to the screen, adjusts their size
     * while keeping them centered, and draws them along with their labels.
     *
     * @param canvas The canvas to draw on.
     */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (boxes.isEmpty() || modelInputWidth <= 1) {
            return
        }

        // The `previewTransformMatrix` correctly maps coordinates from the ImageAnalysis
        // buffer space to the OverlayView's screen space.
        for (box in boxes) {
            // Step 1: Map the detection box from model coordinates to screen coordinates.
            // This gives the full-sized, correctly oriented bounding box.
            val mappedRect = RectF(box.rect)
            previewTransformMatrix.mapRect(mappedRect)

            // Define the scaling factor for the bounding box.
            // 0.5f means half the size (twice as smaller).
            val scaleFactor = 0.65f

            // Calculate the new dimensions, making them twice as smaller.
            val scaledWidth = mappedRect.width() * scaleFactor
            val scaledHeight = mappedRect.height() * scaleFactor

            // Calculate the center of the original mapped box.
            // Apply both vertical and horizontal adjustments here.
            val centerX = mappedRect.centerX() + horizontalAdjustmentPx
            val centerY = mappedRect.centerY() + verticalAdjustmentPx

            // Create the new, scaled rectangle centered within the adjusted position.
            val scaledCenteredRectOnScreen = RectF(
                centerX - (scaledWidth / 2),
                centerY - (scaledHeight / 2),
                centerX + (scaledWidth / 2),
                centerY + (scaledHeight / 2)
            )

            // Draw the scaled and centered bounding box.
            canvas.drawRect(scaledCenteredRectOnScreen, boxPaint)

            // --- Label Text Drawing Logic ---
            // Position the label relative to the scaled and centered box that was just drawn.
            val labelText = "${box.classId}: ${"%.0f".format(box.confidence * 100)}%"
            val textBounds = Rect()
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)

            // Start by positioning the label above the scaled and centered box
            // The text's X position will also be affected by horizontalAdjustmentPx via scaledCenteredRectOnScreen.left
            var textX = scaledCenteredRectOnScreen.left
            var textY = scaledCenteredRectOnScreen.top - textBounds.height() - 12f // 12f padding above box

            // Adjust if the label goes off the top of the screen
            if (textY < 0) {
                textY = scaledCenteredRectOnScreen.bottom + 12f // Draw it below the box instead
            }

            // Adjust if the label goes off the right of the screen
            if (textX + textBounds.width() > width) {
                textX = width.toFloat() - textBounds.width() - 8f // Shift it left
            }
            if (textX < 0) textX = 0f // Ensure it doesn't go off the left

            val textBackgroundRect = RectF(
                textX,
                textY,
                textX + textBounds.width() + 8f,  // Padding for width
                textY + textBounds.height() + 12f // Padding for height
            )

            canvas.drawRect(textBackgroundRect, textBackgroundPaint)
            canvas.drawText(
                labelText,
                textBackgroundRect.left + 4f,
                textBackgroundRect.bottom - 6f,
                textPaint
            )
        }
    }
}