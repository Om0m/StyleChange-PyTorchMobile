package com.omom.imchange

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
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
        imageViewInput.setImageBitmap(scaleChangeBitmap(bitmap, 2.0f))
        imageViewOutput.setImageBitmap(scaleChangeBitmap(bitmap, 2.0f))

        fun transformation(modelName: String) {
            val module = Module.load(assetFilePath(this, modelName))

            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val outputFloatArray = outputTensor.dataAsFloatArray
            val outputByteArray = outputFloatArray.foldIndexed(ByteArray(outputFloatArray.size)) { i, a, v -> a.apply { set(i, v.toByte()) }}

            val bmp = Bitmap.createBitmap(640, 428, Bitmap.Config.ALPHA_8)
            val buffer: ByteBuffer = ByteBuffer.wrap(outputByteArray)
            bmp.copyPixelsFromBuffer(buffer)
            imageViewOutput.setImageBitmap(scaleChangeBitmap(bmp, 2.0f))
        }

        val mosaic = findViewById<Button>(R.id.mosaicButton)
        mosaic.setOnClickListener {
            transformation("mosaic.pt")
        }

        val candy = findViewById<Button>(R.id.candyButton)
        candy.setOnClickListener {
            transformation("candy.pt")
        }

        val udnie = findViewById<Button>(R.id.udnieButton)
        udnie.setOnClickListener {
            transformation("udnie.pt")
        }

        val rain = findViewById<Button>(R.id.rainButton)
        rain.setOnClickListener {
            transformation("rain_princess.pt")
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

private fun scaleChangeBitmap(bitmap: Bitmap, ratio: Float): Bitmap {
    return Bitmap.createScaledBitmap(bitmap, (bitmap.width * ratio).toInt(), (bitmap.height * ratio).toInt(), true)
}
