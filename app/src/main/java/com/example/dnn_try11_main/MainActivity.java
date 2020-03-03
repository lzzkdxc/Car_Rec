package com.example.dnn_try11_main;


import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import androidx.appcompat.app.AppCompatActivity;

import static java.lang.Math.round;
import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;

public class MainActivity extends AppCompatActivity implements OnTouchListener, CvCameraViewListener2 {
    long start = System.currentTimeMillis();
    long end;//获取结束时间
    long end2;//获取结束时间
    void Outtime(){
        try {
            long diff = end - start;//转换为秒数
            long diff2 = end2 - end;//转换为秒数
            System.out.println("time spand : " + diff+"   @@@   "+diff2);
        } catch (Exception e) {
            System.out.println("Got an exception!");
        }
    }
    static {
        OpenCVLoader.initDebug();
    }

    Module module = null, LOGOmodule = null;

    private CameraBridgeViewBase mOpenCvCameraView;
    private ArrayList<String> classes = new ArrayList<>();
    String classesFile = "coco.names";
    //    String modelConfiguration = "/yolov3_1.cfg";
//    String modelWeights = "/latest_plate.weights";
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    int inpWidth = 416;
    int inpHeight = 416;
    Mat frame,dst= new Mat();
    Net net;
    Bitmap bitmap_plate = null, bitmap_logo = null, bitmap_allcar;
    TextView CRNNtextView, LOGOtextView;
    ImageView show;
    int recNu=1;
    ClassCRNN classCRNN=new ClassCRNN();
    ClassLOGO classLOGO=new ClassLOGO();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            String ss = MyUtils.assetFilePath(this, "demo_latest_plate_JIT_CPU.pt");
            classCRNN.module = Module.load(ss);
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            String ss = MyUtils.assetFilePath(this, "DenseNet_car_logo_JIT0221.pt");
            classLOGO.module = Module.load(ss);
        } catch (IOException e) {
            e.printStackTrace();
        }
        mOpenCvCameraView = findViewById(R.id.yolov3cam);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.setOnTouchListener(MainActivity.this);
        mOpenCvCameraView.setMaxFrameSize(1300, 1300);
//        modelConfiguration = Environment.getExternalStorageDirectory().getPath() + modelConfiguration;
//        modelWeights = Environment.getExternalStorageDirectory().getPath() + modelWeights;

        try {
            String s = MyUtils.assetFilePath(this, "YOLO_plate_2class_CPU.weights");
            String ss = MyUtils.assetFilePath(this, "yolov3_2.cfg");
            net = Dnn.readNetFromDarknet(ss, s);
        } catch (IOException e) {
            e.printStackTrace();
        }

        show = findViewById(R.id.image);
//        show.setMaxHeight(show.getMaxWidth());

        CRNNtextView = findViewById(R.id.text);
        LOGOtextView = findViewById(R.id.logo);
        readClasses(classes, classesFile);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);

//        getAppDetailSettingIntent(this);


//        Arrays.sort(MyUtils.classes);
    }

    private void getAppDetailSettingIntent(Context mContext) {
        Intent localIntent = new Intent();
        if (Build.VERSION.SDK_INT >= 9) {
            localIntent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
            localIntent.setData(Uri.fromParts("package", mContext.getPackageName(), null));
        } else if (Build.VERSION.SDK_INT <= 8) {
            localIntent.setAction(Intent.ACTION_VIEW);
            localIntent.setClassName("com.android.settings", "com.android.settings.InstalledAppDetails");
            localIntent.putExtra("com.android.settings.ApplicationPkgName", mContext.getPackageName());
        }
        startActivity(localIntent);
    }
    @SuppressLint("HandlerLeak")
    private Handler showhandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            show.setImageBitmap((Bitmap) msg.obj);
        }
    };
    @SuppressLint("HandlerLeak")
    private Handler CRNNhandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            CRNNtextView.setText(msg.obj.toString());
        }
    };
    @SuppressLint("HandlerLeak")
    private Handler LOGOhandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            LOGOtextView.setText(msg.obj.toString());
        }
    };

    private void readClasses(ArrayList<String> classes, String file) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(getAssets().open(file)));

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes.add(mLine);
            }
        } catch (IOException e) {
            //log the exception
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    Bitmap bitmap_small;
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba();

//        bitmap_small=Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(frame, bitmap_small);

        Imgproc.cvtColor(frame, dst, Imgproc.COLOR_BGR2RGB);

//        Bitmap bitmap_allcar = Bitmap.createBitmap(dst.cols(), dst.rows(), Bitmap.Config.ARGB_8888);
//        Utils.matToBitmap(dst, bitmap_allcar);
//        Bitmap bitmap_t = MyUtils.scaleBitmap(bitmap_allcar, inpWidth, inpHeight);
//        Mat inmat=new Mat();
//
//        Utils.bitmapToMat(bitmap_t,inmat);
//                //        System.out.println("Mat blob:"+inmat.cols()+","+inmat.rows()+"Mat dst:"+dst.size());
//
//        Imgproc.cvtColor(inmat, dst, Imgproc.COLOR_BGR2RGB);
        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0, new Size(inpWidth, inpHeight), new Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        List<Mat> outs = new ArrayList<>();
//                        end = System.currentTimeMillis();
        net.forward(outs, getOutputsNames(net));
//                        end2 = System.currentTimeMillis();
//        Outtime();
//                        start = System.currentTimeMillis();
        System.out.println("recNu_MainAAAAAAA="+recNu);
        postprocess(dst, outs,recNu);
        recNu++;
//        new ShowThread().start();

        return dst;
    }


    void postprocess(Mat frame, List<Mat> outs, int nownu) {
        List<Integer> classIds = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Rect> boxes = new ArrayList<>();
        List<Float> objconf = new ArrayList<>();
        for (int i = 0; i < outs.size(); ++i) {
            for (int j = 0; j < outs.get(i).rows(); ++j) {
                Mat scores = outs.get(i).row(j).colRange(5, outs.get(i).row(j).cols());
                Core.MinMaxLocResult r = Core.minMaxLoc(scores);
                if (r.maxVal > confThreshold) {
                    Mat bb = outs.get(i).row(j).colRange(0, 5);
                    float[] data = new float[1];
                    bb.get(0, 0, data);

                    int centerX = (int) (data[0] * frame.cols());

                    bb.get(0, 1, data);

                    int centerY = (int) (data[0] * frame.rows());

                    bb.get(0, 2, data);

                    int width = (int) (data[0] * frame.cols());

                    bb.get(0, 3, data);

                    int height = (int) (data[0] * frame.rows());

                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    bb.get(0, 4, data);
                    objconf.add(data[0]);

                    confidences.add((float) r.maxVal);
                    classIds.add((int) r.maxLoc.x);
                    boxes.add(new Rect(left, top, width, height));
                }
            }
        }
        MatOfRect boxs = new MatOfRect();

        boxs.fromList(boxes);
        MatOfFloat confis = new MatOfFloat();
        confis.fromList(objconf);
        MatOfInt idxs = new MatOfInt();
        Dnn.NMSBoxes(boxs, confis, confThreshold, nmsThreshold, idxs);
        if (idxs.total() > 0) {
            int[] indices = idxs.toArray();
            for (int i = 0; i < indices.length; ++i) {
                int idx = indices[i];
                Rect box = boxes.get(idx);
                int y = box.y - box.height * 4 < 0 ? 0 : box.y - box.height * 4;
                if (0 == classIds.get(idx)) {
                    Advanced_recognition(box,nownu);
                    drawPred(-1, confidences.get(idx), box.x, y, box.x + box.width, box.y, frame);
                }
                drawPred(classIds.get(idx), confidences.get(idx), box.x, box.y, box.x + box.width, box.y + box.height, frame);
            }
        }

    }
    void Advanced_recognition (Rect box, int nownu){
        int y = box.y - box.height * 4 < 0 ? 0 : box.y - box.height * 4;
        try {
            int w = dst.width(), h = dst.height();
            bitmap_allcar = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dst, bitmap_allcar);
//                        Imgcodecs.imwrite("/storage/sdcard/temp.jpg", frame);
            bitmap_plate = Bitmap.createBitmap(bitmap_allcar, box.x, box.y, box.width, box.height);
            bitmap_logo = Bitmap.createBitmap(bitmap_allcar, box.x, y, box.width, box.y - y);
//                        doCRNN(bmp);
            classCRNN.CRNNgo();
            classLOGO.LOGOgo();
//            if(CRNNrun==null||!CRNNrun.isAlive()){
//                CRNNrun=new CRNNThread(nownu);
//                CRNNrun.start();
//            }
//            if(LOGOrun==null||!LOGOrun.isAlive()){
//                LOGOrun=new LOGOThread(nownu);
//                LOGOrun.start();
//            }

//            new CRNNThread(nownu).start();
//            new LOGOThread(nownu).start();

        } catch (CvException e) {
            Log.d("Exception", e.getMessage());
        }
    }
//    void CRNNgo(){
//        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(1 * 160 * 32);
//        Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_plate, 160, 32);
//        MyUtils.bitmapToFloatBuffer(bmp1, 160, 32, floatBuffer, 0);
//        final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1, 1, 32, 160});
//        assert module != null;
//        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
//        final float[] scores = outputTensor.getDataAsFloatArray();
//        int[] maxx = MyUtils.numMax(scores, 6736);
//        final String ixs = MyUtils.jiema(maxx);
//        String result = MyUtils.quChong(ixs);
//        Message msg = new Message();
//        msg.obj = result;
//        CRNNhandler.sendMessage(msg);
//    }
//    private class CRNNThread extends Thread {
//        private int nownu;
//        public CRNNThread(int nownu)
//        {
//            this.nownu = nownu;
//        }
//        @Override
//        public void run() {
//            try{
//                final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(1 * 160 * 32);
//                Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_plate, 160, 32);
//                MyUtils.bitmapToFloatBuffer(bmp1, 160, 32, floatBuffer, 0);
//                final Tensor inputTensor = Tensor.fromBlob(floatBuffer, new long[]{1, 1, 32, 160});
//                assert module != null;
//                Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
//                final float[] scores = outputTensor.getDataAsFloatArray();
//                int[] maxx = MyUtils.numMax(scores, 6736);
//                final String ixs = MyUtils.jiema(maxx);
//                String result = MyUtils.quChong(ixs);
//                Message msg = new Message();
//                msg.obj = result;
//                CRNNhandler.sendMessage(msg);
//                System.out.println("recNu_CRNNCCC="+nownu);
//            }catch (Exception e) {
//                System.out.println("thread is stop!");
//                e.printStackTrace();
//            }
//
//        }
//    }

    void LOGOgo(){
        Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_logo, 112, 112);
        //            Utile.bitmapToFloatBuffer(bmp1,  112, 112,  floatBuffer, 0);
        //            Utile.bitmapToFloatBuffer(bmp1,0,0,112,112,TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB,);
        //            final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[]{1, 1, 112, 112});
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp1,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //            MyUtils.saveBitmap(Environment.getExternalStorageDirectory().getPath()+"logo",bmp1);
        assert LOGOmodule != null;
        Tensor outputTensor = LOGOmodule.forward(IValue.from(inputTensor)).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        int[] maxx = MyUtils.numMax(scores, 94);
        Message msg = new Message();
        msg.obj = MyUtils.classes[maxx[0]];
        LOGOhandler.sendMessage(msg);
//        System.out.println("recNu_LOGOBBBBB="+nownu);
    }
    private class LOGOThread extends Thread {
        private int nownu;
        public LOGOThread(int nownu)
        {
            this.nownu = nownu;
        }
        @Override
        public void run() {
            try{
//              final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(1 * 112 * 112);
                Bitmap bmp1 = MyUtils.scaleBitmap(bitmap_logo, 112, 112);
    //            Utile.bitmapToFloatBuffer(bmp1,  112, 112,  floatBuffer, 0);
    //            Utile.bitmapToFloatBuffer(bmp1,0,0,112,112,TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB,);
    //            final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[]{1, 1, 112, 112});
                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bmp1,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    //            MyUtils.saveBitmap(Environment.getExternalStorageDirectory().getPath()+"logo",bmp1);
                assert LOGOmodule != null;
                Tensor outputTensor = LOGOmodule.forward(IValue.from(inputTensor)).toTensor();
                final float[] scores = outputTensor.getDataAsFloatArray();
                int[] maxx = MyUtils.numMax(scores, 94);
                Message msg = new Message();
                msg.obj = MyUtils.classes[maxx[0]];
                LOGOhandler.sendMessage(msg);
                System.out.println("recNu_LOGOBBBBB="+nownu);
            }catch (Exception e) {
                System.out.println("thread is stop!");
                e.printStackTrace();
            }
        }
    }
    private class ShowThread extends Thread {
        @Override
        public void run() {
            Message msg = new Message();
            msg.obj = bitmap_allcar;
            showhandler.sendMessage(msg);
        }
    }

    private void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat frame) {
        //Draw a rectangle displaying the bounding box
        rectangle(frame, new Point(left, top), new Point(right, bottom), new Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        String label = String.format("%.2f", conf);
        if (classes.size() > 0) {
            if(classId==-1){
                label="";
            }else{
                label = classes.get(classId) + ":" + label;
            }
//            System.out.println(label);
        }

        //Display the label at the top of the bounding box
        int[] baseLine = new int[1];
        Size labelSize = Imgproc.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
        top = Math.max(top, (int) labelSize.height);
        rectangle(frame, new Point(left, top - round(1.5 * labelSize.height)),
                new Point(left + round(1.5 * labelSize.width), top + baseLine[0]), new Scalar(255, 255, 255), FILLED);
        putText(frame, label, new Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0, 0, 0), 1);
    }

    List<String> getOutputsNames(Net net) {
        ArrayList<String> names = new ArrayList<>();
        if (names.size() == 0) {
            //Get the indices of the output layers, i.e. the layers with unconnected outputs
            List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
            //get the names of all the layers in the network
            List<String> layersNames = net.getLayerNames();

            // Get the names of the output layers in names
            for (int i = 0; i < outLayers.size(); ++i) {
                String layer = layersNames.get(outLayers.get(i).intValue() - 1);
                names.add(layer);
            }
        }
        return names;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        return false;
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }
}
