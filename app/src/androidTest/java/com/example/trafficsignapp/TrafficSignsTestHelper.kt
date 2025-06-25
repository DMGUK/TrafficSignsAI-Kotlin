package com.example.trafficsignapp

import android.graphics.*

object TrafficSignsTestHelper {
    fun preprocess(bitmap: Bitmap, inputWidth: Int, inputHeight: Int, inputChannels: Int = 3): FloatArray {
        val w = inputWidth
        val h = inputHeight
        val px = IntArray(w * h)
        bitmap.getPixels(px, 0, w, 0, 0, w, h)
        val out = FloatArray(1 * inputChannels * h * w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val i = y * w + x
                val c = px[i]
                val r = ((c shr 16) and 0xFF) / 255f
                val g = ((c shr 8) and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                out[i] = r
                out[w * h + i] = g
                out[2 * w * h + i] = b
            }
        }
        return out
    }

    fun postprocess(
        detections: Array<FloatArray>,
        labels: List<String>,
        confThreshold: Float = 0.25f
    ): Pair<String, Float> {
        var bestConf = 0f
        var bestCls = -1
        for (det in detections) {
            val conf = det[4]
            val clsId = det[5].toInt()
            if (conf > bestConf && conf > confThreshold) {
                bestConf = conf
                bestCls = clsId
            }
        }
        return if (bestCls in labels.indices) {
            labels[bestCls] to bestConf
        } else {
            "Unknown" to 0f
        }
    }
}