#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <cxcore.h>
#include <opencv2/opencv.hpp>

//青色の閾値
#define BLUE_MIN_HLSCOLOR cv::Scalar(130, 40, 70)
#define BLUE_MAX_HLSCOLOR cv::Scalar(200, 160, 200)

//赤色の閾値
#define RED_MIN_HLSCOLOR1 cv::Scalar(0, 40, 110)
#define RED_MAX_HLSCOLOR1 cv::Scalar(5, 240, 200)
#define RED_MIN_HLSCOLOR2 cv::Scalar(240, 40, 110)
#define RED_MAX_HLSCOLOR2 cv::Scalar(255, 240, 200)

//黄色の閾値
#define YELLOW_MIN_HLSCOLOR cv::Scalar(20, 20, 40)
#define YELLOW_MAX_HLSCOLOR cv::Scalar(80, 250, 250)

//囲む円の色
#define CIRCLE_COLOR cv::Scalar(255,0,255)

//使う名前空間の宣言
using namespace std;
using namespace cv;

//映像のウィンドウの名前
char windowNameBallDetect[] = "Ball_detect";


void REDMaskGet(cv::Mat *hls_Img, cv::Mat &REDmask_Img, cv::Point2f &center, float &radius){
    //指定色で2値化
        cv::Mat REDmask_Img2;
        cv::inRange(*hls_Img, RED_MIN_HLSCOLOR1, RED_MAX_HLSCOLOR1, REDmask_Img);
        cv::inRange(*hls_Img, RED_MIN_HLSCOLOR2, RED_MAX_HLSCOLOR2, REDmask_Img2);
        cv::bitwise_or(REDmask_Img, REDmask_Img2, REDmask_Img);

        //モルフォロジー変換でノイズ除去
        
        cv::morphologyEx(
         REDmask_Img,
         REDmask_Img,
         cv::MORPH_OPEN,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

        cv::morphologyEx(
         REDmask_Img,
         REDmask_Img,
         cv::MORPH_CLOSE,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

         //輪郭を抽出
        std::vector< std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(REDmask_Img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        //最大の面積を持つ輪郭を計算
        double max_area = 0;
        int max_area_contour = 0;
        for(int j=0 ; j < contours.size() ; j++){
            double area = contourArea(contours.at(j));
            if(max_area < area){
                max_area = area;
                max_area_contour = j;
            }
        }

        //輪郭に対する最小外接円の中心座標と半径を計算
        if(max_area >= 10 && max_area <= 1e5){
            cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
        }
}

void BLUEMaskGet(cv::Mat *hls_Img, cv::Mat &BLUEmask_Img, cv::Point2f &center, float &radius){
        //指定色で2値化
        cv::inRange(*hls_Img, BLUE_MIN_HLSCOLOR, BLUE_MAX_HLSCOLOR, BLUEmask_Img);        

        //モルフォロジー変換でノイズ除去
        cv::morphologyEx(
         BLUEmask_Img,
         BLUEmask_Img,
         cv::MORPH_OPEN,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

        cv::morphologyEx(
         BLUEmask_Img,
         BLUEmask_Img,
         cv::MORPH_CLOSE,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

         //輪郭を抽出
        std::vector< std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(BLUEmask_Img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        //最大の面積を持つ輪郭を計算
        double max_area = 0;
        int max_area_contour = 0;
        for(int j=0 ; j < contours.size() ; j++){
            double area = contourArea(contours.at(j));
            if(max_area < area){
                max_area = area;
                max_area_contour = j;
            }
        }

        //輪郭に対する最小外接円の中心座標と半径を計算
        if(max_area >= 10 && max_area <= 1e5){
            cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
        }
}

void YELLOWMaskGet(cv::Mat *hls_Img, cv::Mat &YELLOWmask_Img, cv::Point2f &center, float &radius){
        //指定色で2値化
        cv::inRange(*hls_Img, YELLOW_MIN_HLSCOLOR, YELLOW_MAX_HLSCOLOR, YELLOWmask_Img);        

        //モルフォロジー変換でノイズ除去
        cv::morphologyEx(
         YELLOWmask_Img,
         YELLOWmask_Img,
         cv::MORPH_OPEN,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

        cv::morphologyEx(
         YELLOWmask_Img,
         YELLOWmask_Img,
         cv::MORPH_CLOSE,        // モルフォロジー演算の種類
         cv::Mat(),
         cv::Point( -1, -1 ),
         2,                     // 収縮と膨張が適用される回数．
         cv::BORDER_CONSTANT,
         cv::morphologyDefaultBorderValue()
         );

         //輪郭を抽出
        std::vector< std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(YELLOWmask_Img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        
        //最大の面積を持つ輪郭を計算
        double max_area = 0;
        int max_area_contour = 0;
        for(int j=0 ; j < contours.size() ; j++){
            double area = contourArea(contours.at(j));
            if(max_area < area){
                max_area = area;
                max_area_contour = j;
            }
        }

        //輪郭に対する最小外接円の中心座標と半径を計算
        if(max_area >= 10 && max_area <= 1e5){
            cv::minEnclosingCircle(contours.at(max_area_contour), center, radius);
        }
}

int main(){

    //使うカメラの宣言
    cv::VideoCapture cap(0);

    //カメラが見つからなかったときの処理
	if ( !cap.isOpened() )
	{
        cout << "[ Error ] Not found camera" << endl;
		return -1;
	}


    //メイン処理
    while( 1 ){

        cv::Mat input_Img;      //フィルタに入力する画像配列
        cv::Mat output_Img;     //フィルタリングされた画像配列
	    cap >> input_Img;       //カメラからの入力動画像をinput_Imgに格納

        //ガウシアンフィルタの適用カーネルサイズは,ksizeで適用
        cv::Size ksize = cv::Size(5, 5);
        cv::GaussianBlur(input_Img, output_Img, ksize, 0);
        //cv::bilateralFilter(input_Img, output_Img, 5, 5, 10);   //処理が重いため,処理能力が高い場合に使用

        //HLS_FULLに変換，FULLの場合Hの範囲が違う
        cv::Mat hls_Img;    //HLS色空間に変換した画像のの格納配列
        cv::cvtColor(output_Img, hls_Img, cv::COLOR_BGR2HLS_FULL);

        cv::Mat YELLOWmask_Img;
        cv::Mat REDmask_Img;
        cv::Mat BLUEmask_Img;
        float radius = 0;
        cv::Point2f center;

        REDMaskGet(&hls_Img, REDmask_Img, center, radius);
        cv::Point2f REDcenter = center;
        int REDradius = radius;
        
        BLUEMaskGet(&hls_Img, BLUEmask_Img, center, radius);
        cv::Point2f BLUEcenter = center;
        int BLUEradius = radius;

        YELLOWMaskGet(&hls_Img, YELLOWmask_Img, center, radius);
        cv::Point2f YELLOWcenter = center;
        int YELLOWradius = radius;

        //2値化画像と元の画像の論理積をとり，指定色のみを残した画像を取得
        cv::Mat masked_Img;
        //cv::bitwise_and(input_Img, input_Img, REDmasked_Img, REDmask_Img);
        //cv::bitwise_and(input_Img, input_Img, BLUEmasked_Img, BLUEmask_Img);
        //cv::bitwise_and(input_Img, input_Img, YELLOWmasked_Img, YELLOWmask_Img);
        
        //輪郭に対する最小外接円を描画
        cv::circle(input_Img, REDcenter, REDradius, CIRCLE_COLOR, 2);
        std::ostringstream REDoss;
        REDoss << REDcenter;
        cv::putText(
            input_Img,
            REDoss.str().c_str(),
            cv::Point(25,75),
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(255,255,255),
            3
        );

        //輪郭に対する最小外接円を描画
        cv::circle(input_Img, BLUEcenter, BLUEradius, CIRCLE_COLOR, 2);
        std::ostringstream BLUEoss;
        BLUEoss << BLUEcenter;
        cv::putText(
            input_Img,
            BLUEoss.str().c_str(),
            cv::Point(25,75),
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(255,255,255),
            3
        );

        //輪郭に対する最小外接円を描画
        cv::circle(input_Img, YELLOWcenter, YELLOWradius, CIRCLE_COLOR, 2);
        std::ostringstream YELLOWoss;
        YELLOWoss << YELLOWcenter;
        cv::putText(
            input_Img,
            YELLOWoss.str().c_str(),
            cv::Point(25,75),
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(255,255,255),
            3
        );

        //指定色のみを残した画像を出力
        cv::imshow(windowNameBallDetect, input_Img);

        //プログラムの終了処理
        int key = cv::waitKey(1);
        if (key == 113)//qボタンが押されたとき
		    {
			    break; //whileループから抜ける．
		    }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}