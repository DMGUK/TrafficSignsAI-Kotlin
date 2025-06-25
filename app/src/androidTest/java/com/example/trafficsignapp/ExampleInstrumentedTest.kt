package com.example.trafficsignapp

import android.graphics.Bitmap
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.trafficsignapp.TrafficSignsTestHelper
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {

    @Test
    fun useAppContext() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.trafficsignapp", appContext.packageName)
    }

    @Test
    fun testLabelFileLoadFromAssets() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val labels = appContext.assets.open("labels.txt").bufferedReader().useLines { it.toList() }
        assertTrue("Label list should not be empty", labels.isNotEmpty())
        assertTrue("First label should be a string", labels[0].isNotBlank())
    }

    @Test
    fun testPostprocessReturnsCorrectLabel() {
        val detections = arrayOf(floatArrayOf(100f, 200f, 50f, 50f, 0.9f, 1f))
        val labels = listOf("China_Stop", "Japan_t-junction")
        val result = TrafficSignsTestHelper.postprocess(detections, labels)
        assertEquals("Japan_t-junction", result.first)
        assertEquals(0.9f, result.second)
    }

    @Test
    fun testPreprocessReturnsCorrectSize() {
        val width = 2
        val height = 2
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val result = TrafficSignsTestHelper.preprocess(bitmap, width, height)
        assertEquals(12, result.size) // 2*2*3 = 12
    }
}
