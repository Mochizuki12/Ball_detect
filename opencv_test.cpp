#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(){
	const char* windowName = "Image";

	cv::VideoCapture cap(0);
	if ( !cap.isOpened() )
	{
		return -1;
	}

	while (1)
	{
		cv::Mat img;
		cap >> img;

		// 必要に応じてここに画像処理

		cv::imshow(windowName, img);//画像を表示．

		int key = cv::waitKey(1);
		if (key == 113)//qボタンが押されたとき
		{
			break;//whileループから抜ける．
		}
	}
	cv::destroyAllWindows();

	return 0;
}
