//////////////////////////////////////////////////////////////////////////
// Copyright (C) 2015-2016 Zhihua Ban. All rights reserved.
// Contact : sawpara@126.com
// Time : 2017/12/18 14:15:10
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include <mex.h>
#include <matrix.h>

#include "supervoxels.hpp"
#include "tools.hpp"


void mat2cv(cv::Mat_<Vec3b> &ou, uint8_t *im, int W, int H){
  ou.create(H, W);
  for (int x = 0; x < W; x++){
    for (int y = 0; y < H; y++){
      uint8_t pixR = im[y + x*H + 0 * W*H];
      uint8_t pixG = im[y + x*H + 1 * W*H];
      uint8_t pixB = im[y + x*H + 2 * W*H];
      ou(y, x) = Vec3b(pixB, pixG, pixR);
    }
  }
}

void cvtmatl(double *ol, vector<int> &il, int W, int H){
  for (int x = 0; x < W; x++){
    for (int y = 0; y < H; y++){
      ol[y + x*H] = il[y*W + x];
    }
  }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
  // read image
  if (nrhs < 2){
    mexPrintf("Usage : \n");
    mexPrintf("\tlabels = mex_vsp(frame_first, frame_second)\n");
    mexPrintf("\tlabels = mex_vsp(frame_first, frame_second, v_s)\n");
    mexPrintf("\tlabels = mex_vsp(frame_first, frame_second, v_x, v_y)\n");
	mexPrintf("\tlabels = mex_vsp(frame_first, frame_second, v_x, v_y, T)\n");
    return;
  }

  // check type
  if (mxGetClassID(prhs[0]) != mxUINT8_CLASS || mxGetClassID(prhs[1]) != mxUINT8_CLASS){
    mexErrMsgTxt("Ban needs two uint8 three channel images");
  }

  uint8_t *frame_ptrs[2]; 
  frame_ptrs[0] = (uint8_t*)mxGetData(prhs[0]);
  frame_ptrs[1] = (uint8_t*)mxGetData(prhs[1]);

  const mwSize  ndim = mxGetNumberOfDimensions(prhs[0]);
  const mwSize *dims = mxGetDimensions(prhs[0]);
  if (ndim != 3 || dims[2] != 3){
    mexErrMsgTxt("input image must be RGB color image!");
  }
  int W = dims[1];
  int H = dims[0];

  
  cv::Mat_<Vec3b> image_1; mat2cv(image_1, frame_ptrs[0], W, H);
  cv::Mat_<Vec3b> image_2; mat2cv(image_2, frame_ptrs[0], W, H);

  int v_x = 10;
  int v_y = 10;
  int T = 20;

  float lambda_c = 100;
  float lambda_s = 2;

  int eta_x = 2;
  int eta_y = 2;

  float sl, sa, sb;
  sl = sa = sb = 10.;
  

  if (nrhs >= 3){
    v_x = v_y = mxGetPr(prhs[2])[0];
  }
  if (nrhs >= 4){
    v_y = mxGetPr(prhs[3])[0];
  }
  if (nrhs >= 5){
    T = mxGetPr(prhs[4])[0];
  }
  
  vector<int> L1;
  vector<int> L2;
  vGMMSP(L1, L2, cv::Mat_<cv::Vec3f>(image_1), cv::Mat_<cv::Vec3f>(image_2), T, v_x, v_y, lambda_s, lambda_c, eta_x, eta_y, sl, sa, sb);

  plhs[0] = mxCreateCellMatrix(2, 1);

  mxArray* lo1 = mxCreateDoubleMatrix(H, W, mxREAL);
  mxArray* lo2 = mxCreateDoubleMatrix(H, W, mxREAL);
  
  cvtmatl(mxGetPr(lo1), L1, W, H);
  cvtmatl(mxGetPr(lo2), L2, W, H);

  mxSetCell(plhs[0], 0, lo1);
  mxSetCell(plhs[0], 1, lo2);
}

