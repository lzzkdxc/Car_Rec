package com.example.dnn_try11_main;

import android.graphics.Bitmap;
import android.util.Log;

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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.round;
import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;

class ClassYOLO {
    Net net;
    private int inpWidth = 416;
    private int inpHeight = 416;
    private float confThreshold = 0.3f;
    private float nmsThreshold = 0.45f;
    private ArrayList<String> classes = new ArrayList<>();
    String classesFile = "coco.names";
    void init(){
        readClasses(classes, classesFile);
        net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
    }
    Mat Go(Mat dst,Mat re){

        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0, new Size(inpWidth, inpHeight), new Scalar(0, 0, 0), true, false);


//        Mat re=dst.clone();
        net.setInput(blob);
        List<Mat> outs = new ArrayList<>();
        end = System.currentTimeMillis();
        net.forward(outs, getOutputsNames(net));
        end2 = System.currentTimeMillis();
//        Outtime();
        start = System.currentTimeMillis();
//        System.out.println("recNu_MainAAAAAAA="+recNu);
        postprocess(re, outs);
        return re;

    }
    private void postprocess(Mat frame, List<Mat> outs) {
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
            for (int idx : indices) {
                Rect box = boxes.get(idx);
                int y = box.y - box.height * 4 < 0 ? 0 : box.y - box.height * 4;
                if (0 == classIds.get(idx)) {
                    MainActivity.getMainActivity().Advanced_Plate_recognition(box);
                    drawPred(-1, confidences.get(idx), box.x, y, box.x + box.width, box.y, frame);
                }else if(1 == classIds.get(idx)){
                    MainActivity.getMainActivity().Advanced_Car_recognition(box);
                }
                drawPred(classIds.get(idx), confidences.get(idx), box.x, box.y, box.x + box.width, box.y + box.height, frame);
            }
        }

    }

    private List<String> getOutputsNames(Net net) {
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

    private long start = System.currentTimeMillis();
    private long end;//获取结束时间
    long end2;//获取结束时间
    private void Outtime(){
        try {
            long diff = end - start;//转换为秒数
            long diff2 = end2 - end;//转换为秒数
            System.out.println("time spand : " + diff+"   @@@   "+diff2);
        } catch (Exception e) {
            System.out.println("Got an exception!");
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
            System.out.println(label);
        }

        //Display the label at the top of the bounding box
        int[] baseLine = new int[1];
        Size labelSize = Imgproc.getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);
        top = Math.max(top, (int) labelSize.height);
        rectangle(frame, new Point(left, top - round(1.5 * labelSize.height)),
                new Point(left + round(1.5 * labelSize.width), top + baseLine[0]), new Scalar(255, 255, 255), FILLED);
        putText(frame, label, new Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, new Scalar(0, 0, 0), 1);
    }
    private void readClasses(ArrayList<String> classes, String file) {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(MyUtils.context.getAssets().open(file)))) {

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                classes.add(mLine);
            }
        } catch (IOException e) {
            //log the exception
        }
        //log the exception
    }
}
