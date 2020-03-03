package com.example.dnn_try11_main;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Environment;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;

public class MyUtils {
    public static String jiema(int[] ixs) {
        String s=alphabets.alphabet;
        String out= "";
        int x=0;
        for(int i=0;i<ixs.length;i++){
            x=ixs[i];
            if(x==0){
                x++;
            }
            out = out + s.charAt(x-1);
        }
        return out;
    }
    public static String  quChong(String input)
    {
        String out="";
        for(int i=0;i<input.length();i++){
            if(i==0||(input.charAt(i)!=input.charAt(i-1)&&input.charAt(i)!='某'))
            {
                out=out+(input.charAt(i));
            }
        }
        return out;
    }
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    public static Bitmap scaleBitmap(Bitmap origin, int newWidth, int newHeight) {
        if (origin == null) {
            return null;
        }
        int height = origin.getHeight();
        int width = origin.getWidth();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);// 使用后乘
        Bitmap newBM = Bitmap.createBitmap(origin, 0, 0, width, height, matrix, false);
        if (!origin.isRecycled()) {
            origin.recycle();
        }
        return newBM;
    }
    public static int[] numMax(float[] a,int length) {
        float value=-Float.MAX_VALUE;
        int[] ixs = new int[41];
        Arrays.fill(ixs, -1);
        int j=0,now=0;
        for (int i = 0; i < a.length; i++) {
            if(j<length){
                if(a[i]>value){
                    value=a[i];
                    ixs[now]=j;
                }
                j++;
            }
            else {
                value=a[i];
                j = 0;
                now++;
                ixs[now]=j;
                i--;
            }
        }
        return ixs;
    }
    public static void bitmapToFloatBuffer(
            final Bitmap bitmap,
            final int width,
            final int height,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {

        final int pixelsCount = height * width;
        final int[] pixels = new int[pixelsCount];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        for (int i = 0; i < pixelsCount; i++) {
            final int c = pixels[i];
            outBuffer.put(outBufferOffset + i, c);
        }
    }
    public static String saveBitmap(String savePath, Bitmap mBitmap) {
        File filePic;
        try {
            filePic = new File(savePath  + ".jpg");
            if (!filePic.exists()) {
                filePic.getParentFile().mkdirs();
                filePic.createNewFile();
            }
            FileOutputStream fos = new FileOutputStream(filePic);
            mBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            return null;
        }

        return filePic.getAbsolutePath();
    }

}
