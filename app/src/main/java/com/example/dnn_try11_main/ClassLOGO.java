package com.example.dnn_try11_main;

import android.graphics.Bitmap;
import android.os.Message;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class ClassLOGO {
    Module module = null;
    Bitmap bitmap_plate;

    String LOGOgo(){
        Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_plate, 112, 112);
        //            Utile.bitmapToFloatBuffer(bmp1,  112, 112,  floatBuffer, 0);
        //            Utile.bitmapToFloatBuffer(bmp1,0,0,112,112,TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB,);
        //            final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[]{1, 1, 112, 112});
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp1,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //            MyUtils.saveBitmap(Environment.getExternalStorageDirectory().getPath()+"logo",bmp1);
        assert module != null;
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        int[] maxx = MyUtils.numMax(scores, 94);
        return Logo_classes.classes[maxx[0]];
//        System.out.println("recNu_LOGOBBBBB="+nownu);
    }
}
