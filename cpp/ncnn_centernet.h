#pragma once
#include<vector>
#include<iostream>
#include<algorithm>
#include<numeric>
#include "net.h"

#define NMS_UNION 1
#define NMS_MIN  2

typedef struct ObjInfo {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;
	int label;
};

class Centerobj {
public:
	Centerobj();
	~Centerobj();

	int init(std::string model_path);


	//You can change the shape of input image by setting params :resized_w and resized_h
	int detect(ncnn::Mat &inblob, std::vector<ObjInfo>&objs, int resized_w,int resized_h,
		float scoreThresh = 0.2, float nmsThresh = 0.45);

private:
	void dynamicScale(float in_w, float in_h);
	void genIds(float * heatmap, int h, int w,int c, float thresh, std::vector<float> &ids);
	void nms(std::vector<ObjInfo>& input, std::vector<ObjInfo>& output, float nmsthreshold = 0.3,int type=NMS_MIN);
	void decode(ncnn::Mat & heatmap  , ncnn::Mat &scale, ncnn::Mat &offset,
		std::vector<ObjInfo>&objs, float scoreThresh, float nmsThresh);
private:
	ncnn::Net net;

	int d_h;
	int d_w;
	float d_scale_h;
	float d_scale_w;

	float scale_h;
	float scale_w;

	int image_h;
	int image_w;
};


