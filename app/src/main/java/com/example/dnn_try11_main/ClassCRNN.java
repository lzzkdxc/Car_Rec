package com.example.dnn_try11_main;

import android.graphics.Bitmap;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.nio.FloatBuffer;

class ClassCRNN {
    Module module = null;
    Bitmap bitmap_plate;
    TextView textView;
    void CRNNgo(){
        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer( 160 * 32);
        Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_plate, 160, 32);
        MyUtils.bitmapToFloatBuffer(bmp1, 160, 32, floatBuffer, 0);
        final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1, 1, 32, 160});
        assert module != null;
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        int[] maxx = MyUtils.numMax(scores, 6736);
        final String ixs = jiema(maxx);
        String result = MyUtils.quChong(ixs);
        MyUtils.showTextView(textView,result);
    }
    private String jiema(int[] ixs) {
        StringBuilder out= new StringBuilder();
        int x;
        for (int ix : ixs) {
            x = ix;
            if (x == 0) {
                x++;
            }
            out.append(MyUtils.alphabet.charAt(x - 1));
        }
        return out.toString();
    }
}
