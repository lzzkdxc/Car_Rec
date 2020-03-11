package com.example.dnn_try11_main;

import android.graphics.Bitmap;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

class ClassTYPE {
    Module module = null;
    Bitmap bitmap_plate;
    TextView textView;
    void TYPEgo(){
        Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_plate, 224, 224);
        assert bmp1 != null;
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp1,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //            MyUtils.saveBitmap(Environment.getExternalStorageDirectory().getPath()+"logo",bmp1);
        assert module != null;
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        final float[] scores = outputTensor.getDataAsFloatArray();
        int maxx = f(scores);
        MyUtils.showTextView(textView,MyUtils.TYPEclasses[maxx]);
//        System.out.println("recNu_LOGOBBBBB="+nownu);
    }
    int f(float[] a){
        float max=Float.MAX_VALUE;
        int maxnu=0;
        for (int i = 0; i < a.length; i++) {
            if(a[i]>max){
                max=a[i];
                maxnu=i;
            }
        }
        return maxnu;
    }
}
