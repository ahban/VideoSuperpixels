#ifndef __VGMMSP_HPP__
#define __VGMMSP_HPP__

#include <cmath>
#include <algorithm>
#include <vector>
#include "su/matrix.hpp"
#include <opencv2/opencv.hpp>

using namespace std;


void random_colors(vector<cv::Vec3b> &rdcolors, int K){
  rdcolors.resize(K);
  for (int k = 0; k < K; k++){
    rdcolors[k] = cv::Vec3b(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
  }
}

cv::Mat paint(int W, int H, const vector<int> &labels, const vector<cv::Vec3b> &colors){
  cv::Mat_<cv::Vec3b> image_painted(H, W);

  for (int y = 0; y < H; y++){
    for (int x = 0; x < W; x++){
      int k = labels[y*W + x];
      image_painted(y, x) = colors[k];
    }
  }
  return image_painted;
}


struct Theta
{
  // mu_k
  float mx1;
  float my1;
  float mx2;
  float my2;
  float ml, ma, mb;

  // Sigma_k
  float isx1, isy1;
  float vx1y1_00, vx1y1_01;
  float vx1y1_10, vx1y1_11;

  float isx2, isy2;
  float vx2y2_00, vx2y2_01;
  float vx2y2_10, vx2y2_11;

  float isl;

  float isa, isb;
  float vab_00, vab_01;
  float vab_10, vab_11;


  float sigma_idet1;
  float sigma_idet2;

  int XXB, XXE, YYB, YYE;
};

inline void tDNSymE2x2(float a00, float a01, float a11, float &e0, float &e1, float &v00, float &v01, float &v10, float &v11){
  float const zero = (float)0, one = (float)1, half = (float)0.5;
  float c2 = half * (a00 - a11), s2 = a01;
  float maxAbsComp = max(abs(c2), abs(s2));
  if (maxAbsComp > zero){
    c2 /= maxAbsComp;  // in [-1,1]
    s2 /= maxAbsComp;  // in [-1,1]
    float length = sqrt(c2 * c2 + s2 * s2);
    c2 /= length;
    s2 /= length;
    if (c2 > zero){
      c2 = -c2;	s2 = -s2;
    }
  }
  else{
    c2 = -one; s2 = zero;
  }
  float s = sqrt(half * (one - c2));  // >= 1/sqrt(2)
  float c = half * s2 / s;

  float csqr = c * c, ssqr = s * s, mid = s2 * a01;
  e0 = csqr * a00 + mid + ssqr * a11;
  e1 = csqr * a11 - mid + ssqr * a00;

  v00 = c;	v01 = s;
  v10 = -s; 	v11 = c;
}

static void vGMMSP_extract_labels(
  vector<int> &L1,
  vector<int> &L2,
  const su::Mat<float> &R1,
  const su::Mat<float> &R2,
  const vector<Theta> &theta,
  const int v_x,
  const int v_y,
  const int n_x,
  const int n_y,
  const int t_x,
  const int t_y,
  const int W,
  const int H
  ){
  L1.resize(W*H);
  L2.resize(W*H);
  int rl = (W - n_x*v_x) >> 1;
  int ru = (H - n_y*v_y) >> 1;
  int hxs = v_x >> 1;
  int hys = v_y >> 1;

  for (int y = 0; y < H; y++){
    for (int x = 0; x < W; x++){

      int ilabel_x = (x - rl) / v_x; if (ilabel_x == n_x) ilabel_x = n_x - 1;
      int ilabel_y = (y - ru) / v_y; if (ilabel_y == n_y) ilabel_y = n_y - 1;

      float max_dense_1 = -FLT_MAX;
      float max_dense_2 = -FLT_MAX;
      int final_label_1 = -1;
      int final_label_2 = -1;

      for (int dy = -t_y; dy <= t_y; dy++){
        for (int dx = -t_x; dx <= t_x; dx++){
          const int al_x = ilabel_x + dx;
          const int al_y = ilabel_y + dy;
          if (al_x < 0 || al_y < 0 || al_x >= n_x || al_y >= n_y){
            continue;
          }
          const int al_k = al_y*n_x + al_x;

          float cur_dense;
          cur_dense = R1(y - theta[al_k].YYB, x - theta[al_k].XXB, al_k);///?????
          if (max_dense_1 < cur_dense){
            max_dense_1 = cur_dense;
            final_label_1 = al_k;
          }
          cur_dense = R2(y - theta[al_k].YYB, x - theta[al_k].XXB, al_k);///?????
          if (max_dense_2 < cur_dense){
            max_dense_2 = cur_dense;
            final_label_2 = al_k;
          }
        }
      }

      L1[y*W + x] = final_label_1;
      L2[y*W + x] = final_label_2;
    }
  }
}


static void vGMMSP_update_theta(
  vector<Theta> &theta,
  const cv::Mat_<cv::Vec3f> &f1, // OpenCV frame t
  const cv::Mat_<cv::Vec3f> &f2, // OpenCV frame t+1
  const su::Mat<float> &R1,
  const su::Mat<float> &R2,
  const int v_x,
  const int v_y,
  const int n_x,
  const int n_y,
  const int t_x,
  const int t_y,
  const float e_s,
  const float e_c
){
  // for each superpixel
#pragma omp parallel for
  for (int k_y = 0; k_y < n_y; k_y++){
    for (int k_x = 0; k_x < n_x; k_x++){
      int k = k_x + k_y*n_x;

      //////////////////////////////////////////////////////////////////////////
      // mu
      float mx1 = 0, my1 = 0, md1 = 0;
      float mx2 = 0, my2 = 0, md2 = 0, ml = 0, ma = 0, mb = 0;

      for (int y = theta[k].YYB; y < theta[k].YYE; y++){
        for (int x = theta[k].XXB; x < theta[k].XXE; x++){
          const float RV1 = R1(y - theta[k].YYB, x - theta[k].XXB, k);
          const float RV2 = R2(y - theta[k].YYB, x - theta[k].XXB, k);
          const cv::Vec3f px1 = f1(y, x);
          const cv::Vec3f px2 = f2(y, x);
          mx1 += RV1*x; mx2 += RV2*x;
          my1 += RV1*y; my2 += RV2*y;

          ml += RV1*px1[0]; ml += RV2*px2[0];
          ma += RV1*px1[1]; ma += RV2*px2[1];
          mb += RV1*px1[2]; mb += RV2*px2[2];

          md1 += RV1;
          md2 += RV2;
        }
      }

      if (md1 < 1 || md2 < 1){
        continue;
      }


      mx1 = mx1 / md1; my1 = my1 / md1;
      mx2 = mx2 / md2; my2 = my2 / md2;

      ml = ml / (md1 + md2);
      ma = ma / (md1 + md2);
      mb = mb / (md1 + md2);

      //////////////////////////////////////////////////////////////////////////
      // sigma
      float tp0, tp1;

      float xy1_00 = 0; float xy2_00 = 0; float ab_00 = 0;
      float xy1_01 = 0; float xy2_01 = 0; float ab_01 = 0;
      float xy1_11 = 0; float xy2_11 = 0; float ab_11 = 0;
      float sl = 0;

      for (int y = theta[k].YYB; y < theta[k].YYE; y++){
        for (int x = theta[k].XXB; x < theta[k].XXE; x++){

          const float RV1 = R1(y - theta[k].YYB, x - theta[k].XXB, k);
          const float RV2 = R2(y - theta[k].YYB, x - theta[k].XXB, k);

          const cv::Vec3f px1 = f1(y, x);
          const cv::Vec3f px2 = f2(y, x);

          // f1
          tp0 = x - mx1; tp1 = y - my1;
          xy1_00 += RV1 * tp0 * tp0;
          xy1_01 += RV1 * tp0 * tp1;
          xy1_11 += RV1 * tp1 * tp1;

          // f2
          tp0 = x - mx2; tp1 = y - my2;
          xy2_00 += RV2 * tp0 * tp0;
          xy2_01 += RV2 * tp0 * tp1;
          xy2_11 += RV2 * tp1 * tp1;

          // l
          tp0 = px1[0] - ml;
          sl += RV1 * tp0 * tp0;
          tp0 = px2[0] - ml;
          sl += RV2 * tp0 * tp0;

          // ab
          tp0 = px1[1] - ma;
          tp1 = px1[2] - mb;
          ab_00 += RV1 * tp0 * tp0;
          ab_01 += RV1 * tp0 * tp1;
          ab_11 += RV1 * tp1 * tp1;

          tp0 = px2[1] - ma;
          tp1 = px2[2] - mb;
          ab_00 += RV2 * tp0 * tp0;
          ab_01 += RV2 * tp0 * tp1;
          ab_11 += RV2 * tp1 * tp1;

        }
      }

      xy1_00 = xy1_00 / md1; xy1_01 = xy1_01 / md1; xy1_11 = xy1_11 / md1;
      xy2_00 = xy2_00 / md2; xy2_01 = xy2_01 / md2; xy2_11 = xy2_11 / md2;
      ab_00 = ab_00 / (md1 + md2);
      ab_01 = ab_01 / (md1 + md2);
      ab_11 = ab_11 / (md1 + md2);
      sl = sl / (md1 + md2);

      float isx1, isy1, vxy1_00, vxy1_01, vxy1_10, vxy1_11;
      float isx2, isy2, vxy2_00, vxy2_01, vxy2_10, vxy2_11;
      float isa, isb, vab_00, vab_01, vab_10, vab_11;
      float isl;

      tDNSymE2x2(xy1_00, xy1_01, xy1_11, isx1, isy1, vxy1_00, vxy1_01, vxy1_10, vxy1_11);
      if (isx1 < e_s) { isx1 = e_s; } isx1 = 1. / isx1;
      if (isy1 < e_s) { isy1 = e_s; } isy1 = 1. / isy1;

      tDNSymE2x2(xy2_00, xy2_01, xy2_11, isx2, isy2, vxy2_00, vxy2_01, vxy2_10, vxy2_11);
      if (isx2 < e_s) { isx2 = e_s; } isx2 = 1. / isx2;
      if (isy2 < e_s) { isy2 = e_s; } isy2 = 1. / isy2;

      tDNSymE2x2(ab_00, ab_01, ab_11, isa, isb, vab_00, vab_01, vab_10, vab_11);
      if (isa < e_c) { isa = e_c; } isa = 1. / isa;
      if (isb < e_c) { isb = e_c; } isb = 1. / isb;

      if (sl < e_c){ sl = e_c; } isl = 1. / sl;

      theta[k].mx1 = mx1;
      theta[k].my1 = my1;
      theta[k].mx2 = mx2;
      theta[k].my2 = my2;
      theta[k].ml = ml;
      theta[k].ma = ma;
      theta[k].mb = mb;

      theta[k].isx1 = isx1;
      theta[k].isy1 = isy1;

      theta[k].isx2 = isx2;
      theta[k].isy2 = isy2;

      theta[k].isl = isl;
      theta[k].isa = isa;
      theta[k].isb = isb;

      theta[k].vx1y1_00 = vxy1_00;
      theta[k].vx1y1_01 = vxy1_01;
      theta[k].vx1y1_10 = vxy1_10;
      theta[k].vx1y1_11 = vxy1_11;

      theta[k].vx2y2_00 = vxy2_00;
      theta[k].vx2y2_01 = vxy2_01;
      theta[k].vx2y2_10 = vxy2_10;
      theta[k].vx2y2_11 = vxy2_11;

      theta[k].vab_00 = vab_00;
      theta[k].vab_01 = vab_01;
      theta[k].vab_10 = vab_10;
      theta[k].vab_11 = vab_11;

      theta[k].sigma_idet1 = sqrt(isl * isa * isb * isx1 * isy1);
      theta[k].sigma_idet2 = sqrt(isl * isa * isb * isx2 * isy2);

    }
  }


}


static void vGMMSP_update_R(
  su::Mat<float> &R1,
  su::Mat<float> &R2,
  const cv::Mat_<cv::Vec3f> &f1, // OpenCV frame t
  const cv::Mat_<cv::Vec3f> &f2, // OpenCV frame t+1
  const vector<Theta> &theta,
  const int v_x,
  const int v_y,
  const int n_x,
  const int n_y,
  const int t_x,
  const int t_y
  ){
#define epsilon_t (FLT_MIN*9.f)
  int W = f1.cols;
  int H = f1.rows;

  int rl = (W - n_x*v_x) >> 1;
  int ru = (H - n_y*v_y) >> 1;
  int hxs = v_x >> 1;
  int hys = v_y >> 1;

  int ntx = 2 * t_x + 1;
  int nty = 2 * t_y + 1;

  // for each i
#pragma omp parallel 
  {
    vector<float> tpR1(ntx*nty);
    vector<float> tpR2(ntx*nty);
#pragma omp for
    for (int y = 0; y < H; y++){
      for (int x = 0; x < W; x++){
        cv::Vec3f pix1 = f1(y, x);
        cv::Vec3f pix2 = f2(y, x);

        int ik_x = (x - rl) / v_x; if (ik_x >= n_x) ik_x = n_x - 1;
        int ik_y = (y - ru) / v_y; if (ik_y >= n_y) ik_y = n_y - 1;

        // K_i
        tpR1.assign(nty*ntx, 0);
        tpR2.assign(nty*ntx, 0);
        float ffi0, ffi1;
        float ffo0, ffo1;
        float ff;
        float d_x1y1, d_x2y2, d_l1, d_l2, d_ab1, d_ab2;
        float D1, D2;
        float sum_R1 = 0;
        float sum_R2 = 0;
        float sum_ok = 0;

        for (int dy = -t_y; dy <= t_y; dy++){
          for (int dx = -t_x; dx <= t_x; dx++){
            int ok_x = ik_x + dx;
            int ok_y = ik_y + dy;
            if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y)
              continue;

            int ok = ok_y * n_x + ok_x;

            // x1 y1
            ffi0 = x - theta[ok].mx1; ffi1 = y - theta[ok].my1;
            ffo0 = ffi0 * theta[ok].vx1y1_00 + ffi1 * theta[ok].vx1y1_01; ffo0 = ffo0 * ffo0;
            ffo1 = ffi0 * theta[ok].vx1y1_10 + ffi1 * theta[ok].vx1y1_11; ffo1 = ffo1 * ffo1;
            d_x1y1 = ffo0 * theta[ok].isx1 + ffo1 * theta[ok].isy1;

            // x2 y2 
            ffi0 = x - theta[ok].mx2; ffi1 = y - theta[ok].my2;
            ffo0 = ffi0 * theta[ok].vx2y2_00 + ffi1 * theta[ok].vx2y2_01; ffo0 = ffo0 * ffo0;
            ffo1 = ffi0 * theta[ok].vx2y2_10 + ffi1 * theta[ok].vx2y2_11; ffo1 = ffo1 * ffo1;
            d_x2y2 = ffo0 * theta[ok].isx2 + ffo1 * theta[ok].isy2;

            // CIE-l-1
            ff = pix1[0] - theta[ok].ml; ff = ff*ff;
            d_l1 = ff * theta[ok].isl;

            // CIE-l-2
            ff = pix2[0] - theta[ok].ml; ff = ff*ff;
            d_l2 = ff * theta[ok].isl;

            // CIE-ab-1
            ffi0 = pix1[1] - theta[ok].ma;
            ffi1 = pix1[2] - theta[ok].mb;
            ffo0 = ffi0 * theta[ok].vab_00 + ffi1 * theta[ok].vab_01; ffo0 = ffo0 * ffo0;
            ffo1 = ffi0 * theta[ok].vab_10 + ffi1 * theta[ok].vab_11; ffo1 = ffo1 * ffo1;
            d_ab1 = ffo0 * theta[ok].isa + ffo1 * theta[ok].isb;

            // CIE-ab-2
            ffi0 = pix2[1] - theta[ok].ma;
            ffi1 = pix2[2] - theta[ok].mb;
            ffo0 = ffi0 * theta[ok].vab_00 + ffi1 * theta[ok].vab_01; ffo0 = ffo0 * ffo0;
            ffo1 = ffi0 * theta[ok].vab_10 + ffi1 * theta[ok].vab_11; ffo1 = ffo1 * ffo1;
            d_ab2 = ffo0 * theta[ok].isa + ffo1 * theta[ok].isb;

            // D1
            D1 = (d_x1y1 + d_l1 + d_ab1)*(-0.5f);
            D2 = (d_x2y2 + d_l2 + d_ab2)*(-0.5f);

            tpR1[(dy + t_y)*ntx + t_x + dx] = exp(D1) * theta[ok].sigma_idet1;
            tpR2[(dy + t_y)*ntx + t_x + dx] = exp(D2) * theta[ok].sigma_idet2;

            sum_R1 += exp(D1) * theta[ok].sigma_idet1;
            sum_R2 += exp(D2) * theta[ok].sigma_idet2;

            sum_ok += 1.f;

          }
        } // end K_i
        if (sum_R1 < epsilon_t){
          for (int dy = -t_y; dy <= t_y; dy++){
            for (int dx = -t_x; dx <= t_x; dx++){
              int ok_x = ik_x + dx;
              int ok_y = ik_y + dy;
              if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y){
                continue;
              }
              int ok = ok_y*n_x + ok_x;
              R1(y - theta[ok].YYB, x - theta[ok].XXB, ok) = float(1) / float(sum_ok);
            }
          }
        }
        else{
          for (int dy = -t_y; dy <= t_y; dy++){
            for (int dx = -t_x; dx <= t_x; dx++){
              int ok_x = ik_x + dx;
              int ok_y = ik_y + dy;
              if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y){
                continue;
              }
              int ok = ok_y*n_x + ok_x;
              R1(y - theta[ok].YYB, x - theta[ok].XXB, ok) = tpR1[(dy + t_y)*ntx + t_x + dx] / sum_R1;
            }
          }
        }

        if (sum_R2 < epsilon_t){
          for (int dy = -t_y; dy <= t_y; dy++){
            for (int dx = -t_x; dx <= t_x; dx++){
              int ok_x = ik_x + dx;
              int ok_y = ik_y + dy;
              if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y){
                continue;
              }
              int ok = ok_y*n_x + ok_x;
              R2(y - theta[ok].YYB, x - theta[ok].XXB, ok) = float(1) / float(sum_ok);
            }
          }
        }
        else{
          for (int dy = -t_y; dy <= t_y; dy++){
            for (int dx = -t_x; dx <= t_x; dx++){
              int ok_x = ik_x + dx;
              int ok_y = ik_y + dy;
              if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y){
                continue;
              }
              int ok = ok_y*n_x + ok_x;
              R2(y - theta[ok].YYB, x - theta[ok].XXB, ok) = tpR2[(dy + t_y)*ntx + t_x + dx] / sum_R2;
            }
          }
        }

      }
    } // end i
  }
#undef epsilon_t
}

static void vGMMSP_init_theta(
  vector<Theta> &theta,
  const cv::Mat_<cv::Vec3f> &f1,
  const cv::Mat_<cv::Vec3f> &f2,
  const int v_x,
  const int v_y,
  const int n_x,
  const int n_y,
  const float sl,
  const float sa,
  const float sb,
  const int W,
  const int H
){

  float isx = 1. / (v_x*v_x); // x
  float isy = 1. / (v_y*v_y); // y
  float isl = 1. / (sl*sl); // l
  float isa = 1. / (sa*sa); // a
  float isb = 1. / (sb*sb); // b

  int rl = (W - n_x*v_x) >> 1;
  int ru = (H - n_y*v_y) >> 1;
  int hxs = v_x >> 1;
  int hys = v_y >> 1;

  Theta tpt;
  tpt.isx1 = isx;
  tpt.isy1 = isy;

  tpt.isx2 = isx;
  tpt.isy2 = isy;

  tpt.isl = isl;
  tpt.isa = isa;
  tpt.isb = isb;

  tpt.vx1y1_00 = 1.; tpt.vx1y1_01 = 0.;
  tpt.vx1y1_10 = 0.; tpt.vx1y1_11 = 1.;

  tpt.vx2y2_00 = 1.; tpt.vx2y2_01 = 0.;
  tpt.vx2y2_10 = 0.; tpt.vx2y2_11 = 1.;

  tpt.vab_00 = 1.; tpt.vab_01 = 0.;
  tpt.vab_10 = 0.; tpt.vab_11 = 1.;

  tpt.sigma_idet1 = sqrt(isx*isy*isl*isa*isb);
  tpt.sigma_idet2 = sqrt(isx*isy*isl*isa*isb);

  for (int k_y = 0; k_y < n_y; k_y++){
    for (int k_x = 0; k_x < n_x; k_x++){
      int k = k_y * n_x + k_x;

      int fx = (rl + k_x*v_x + hxs);
      int fy = (ru + k_y*v_y + hys);

      tpt.mx1 = fx;
      tpt.my1 = fy;

      tpt.mx2 = fx;
      tpt.my2 = fy;

      tpt.ml = 0.5*(f1(fy, fx)[0] + f2(fy, fx)[0]);
      tpt.ma = 0.5*(f1(fy, fx)[1] + f2(fy, fx)[1]);
      tpt.mb = 0.5*(f1(fy, fx)[2] + f2(fy, fx)[2]);

      if ((k_x - 1) <= 0)         tpt.XXB = 0;  else tpt.XXB = (k_x - 1) * v_x + rl;
      if ((k_x + 1) >= (n_x - 1)) tpt.XXE = W;	  else tpt.XXE = (k_x + 2) * v_x + rl;
      if ((k_y - 1) <= 0)         tpt.YYB = 0;  else tpt.YYB = (k_y - 1) * v_y + ru;
      if ((k_y + 1) >= (n_y - 1)) tpt.YYE = H;  	else tpt.YYE = (k_y + 2) * v_y + ru;

      theta[k] = tpt;
    }
  }
  return;
}


static void vGMMSP(vector<int> &L1, vector<int> &L2, const cv::Mat_<cv::Vec3f> &f1, const cv::Mat_<cv::Vec3f> &f2, int T, int v_x, int v_y, float e_s, float e_c, int t_x, int t_y, float sl, float sa, float sb){
  int W = f2.cols;
  int H = f2.rows;

  su::Mat<float> R1;
  su::Mat<float> R2;
  int ntx = 2 * t_x + 1;
  int nty = 2 * t_y + 1;

  int n_x = W / v_x;
  int n_y = H / v_y;

  int K = n_x * n_y;

  int rdw = W - n_x*v_x;
  int rdh = H - n_y*v_y;

  // theta cell size
  int tcw = v_x * ntx + rdw;
  int tch = v_y * nty + rdh;

  R1.create(tch, tcw, K);
  R2.create(tch, tcw, K);

  R1.clear();
  R2.clear();

  vector<Theta> theta(K);

  vGMMSP_init_theta(theta, f1, f2, v_x, v_y, n_x, n_y, sl, sa, sb, W, H);

  vGMMSP_update_R(R1, R2, f1, f2, theta, v_x, v_y, n_x, n_y, t_x, t_y);

  for (int it = 0; it < T; it++){
    vGMMSP_update_theta(theta, f1, f2, R1, R2, v_x, v_y, n_x, n_y, t_x, t_y, e_s, e_c);
    vGMMSP_update_R(R1, R2, f1, f2, theta, v_x, v_y, n_x, n_y, t_x, t_y);
  }
  vGMMSP_extract_labels(L1, L2, R1, R2, theta, v_x, v_y, n_x, n_y, t_x, t_y, W, H);
}

#endif
