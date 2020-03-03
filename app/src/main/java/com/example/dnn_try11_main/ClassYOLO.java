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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import java.util.ArrayList;
import java.util.List;

public class ClassYOLO {
    Net net;
    int inpWidth = 416;
    int inpHeight = 416;
    void init(){

    }
    Mat Go(Mat dst){

        Mat blob = Dnn.blobFromImage(dst, 1 / 255.0, new Size(inpWidth, inpHeight), new Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        List<Mat> outs = new ArrayList<>();
        end = System.currentTimeMillis();
        net.forward(outs, getOutputsNames(net));
        end2 = System.currentTimeMillis();
        Outtime();
        start = System.currentTimeMillis();
//        System.out.println("recNu_MainAAAAAAA="+recNu);
        postprocess(dst, outs);

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
            Bitmap bitmap_allcar = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dst, bitmap_allcar);
//                        Imgcodecs.imwrite("/storage/sdcard/temp.jpg", frame);
            classCRNN.bitmap_plate = Bitmap.createBitmap(bitmap_allcar, box.x, box.y, box.width, box.height);
            classLOGO.bitmap_plate = Bitmap.createBitmap(bitmap_allcar, box.x, y, box.width, box.y - y);
            classCRNN.CRNNgo();
            classLOGO.LOGOgo();

        } catch (CvException e) {
            Log.d("Exception", e.getMessage());
        }
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
}
