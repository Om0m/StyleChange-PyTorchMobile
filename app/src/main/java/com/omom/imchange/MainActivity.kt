package com.omom.imchange

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val bitmap = BitmapFactory.decodeStream(assets.open("akarenga.jpg"))

        val module = Module.load(assetFilePath(this, "mosaic.pt"))

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
        Log.d("MainActivity", outputTensor.toString())
        val outputFloatArray = outputTensor.dataAsFloatArray
        Log.d("MainActivity", outputFloatArray.size.toString())
        val outputByteArray = outputFloatArray.foldIndexed(ByteArray(outputFloatArray.size)) { i, a, v -> a.apply { set(i, v.toByte()) }}
        Log.d("MainActivity", outputByteArray.size.toString())

//        val opt = BitmapFactory.Options()
//        opt.inJustDecodeBounds = false
//        val outputBitmap = BitmapFactory.decodeByteArray(outputByteArray, 0, outputByteArray.size, opt)

        val bmp = Bitmap.createBitmap(640, 426, Bitmap.Config.ALPHA_8)
//        val bmp2 = Bitmap.createBitmap(640, 428, 3, Bitmap.Config.ALPHA_8)
        val buffer: ByteBuffer = ByteBuffer.wrap(outputByteArray)
        bmp.copyPixelsFromBuffer(buffer)

        val button = findViewById<Button>(R.id.transButton)
        button.setOnClickListener {
//            Log.d("MainActivity", outputBitmap.byteCount.toString())
            imageViewOutput.setImageBitmap(bmp)
        }
    }
}

private fun assetFilePath(context: Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (file.exists() && file.length() > 0) {
        return file.absolutePath
    }
    context.assets.open(assetName).use { inputStream ->
        FileOutputStream(file).use { outputStream ->
            val buffer = ByteArray(4 * 1024)
            var read: Int
            while (inputStream.read(buffer).also { read = it } != -1) {
                outputStream.write(buffer, 0, read)
            }
            outputStream.flush()
        }
        return file.absolutePath
    }
}
