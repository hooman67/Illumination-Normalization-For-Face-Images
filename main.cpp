#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <string>
#include <iostream>

#include <cstdio>
#include <cmath>


#include "apLTV.hpp"
#include "apmyDCT.hpp"
#include "apImgProcess.hpp"
#include "apSmallScaleAdjust.hpp"

namespace fs = std::experimental::filesystem;
using namespace cv;

int main()
{

	/********************************************************************************************************
	****************************    HS **********************************************************************
    *********************************************************************************************************/
    std::string path_in = "/home/hooman/Illumination-Normalization-For-Face-Images/dataset";
    std::string path_out = "/home/hooman/Illumination-Normalization-For-Face-Images/output";


	//all data is allcoated upfront
	double* img_d;
	double* largeScale_exp_d;
	double **img_2d;
	double *smallScale_exp_d;
	double **smallScale_exp_2d;
	unsigned char *img_recon_c;
	double **largescale_2d;
	double **smallscale_2d;
    cv::Mat largedct_cvmat;
    cv::Mat img_in;
    // cv::Mat img_out;
    size_t img_in_size_bytes;



    for (const auto & entry : fs::directory_iterator(path_in)){
    	std::string img_in_path_str = entry.path().string();
        std::cout << "\nprocessing Image: " << img_in_path_str << std::endl;

		img_in = cv::imread(img_in_path_str, cv::IMREAD_GRAYSCALE);
		std::cout << "\n\nimg_in width: " << img_in.cols << "\n" <<
		"img_out height: " << img_in.rows << "\n" <<
		"img_out depth: " << img_in.depth() << "\n" <<
		"img_out channels " << img_in.channels() << "\n" <<
		"img_out type: " << img_in.type() << "\n";
		// cv::Mat img_out = img_in.clone();

		img_in_size_bytes = img_in.total() * img_in.elemSize();

		if (img_in.empty()){
			std::cout << "can't open the image skipping it.\n";
			continue;
		}

		if( (img_in.rows * img_in.cols) % 2 != 0){
			std::cout << "can't run apLargeScale_LogDCT on odd dimention images. Skipping it.\n";
			continue;
		}

		
		/*****image decomposition************/
		//log transform, also transforms the data to double
		img_d = apLogUC2(img_in.data, img_in_size_bytes);

		//matrix transform to 2d
		img_2d = ap1DTo2Dd(img_d, img_in.rows, img_in.cols);

		//matrix transform back to 1d
		largescale_2d = apLTV(img_2d, img_in.rows, img_in.cols, 0.4, 0.1, 100);
		smallscale_2d = apGetSmallScale(img_2d, largescale_2d, img_in.rows, img_in.cols);


		/*************image process******************/
		// normalizing illumination
		largedct_cvmat = apLargeScale_LogDCT(img_in.rows, img_in.cols, largescale_2d); // normalize large-scale image


		/************image reconstruction*************/
		//conduct exponential transform
		largeScale_exp_d = apExp1D((double*)largedct_cvmat.data, img_in_size_bytes);//large
		
		//conduct exponential transform
		smallScale_exp_2d = apExp2D(smallscale_2d, img_in.rows, img_in.cols);//small
		smallScale_exp_d = ap2DTo1Dd(smallScale_exp_2d, img_in.rows, img_in.cols);


		for (int i = 0; i < img_in_size_bytes; i++){
			img_d[i] = largeScale_exp_d[i] * smallScale_exp_d[i];
		}

		//transform double data to char data with 100 times strengthened
		img_recon_c = apDtoC_strengthened(img_d, img_in_size_bytes, 100);


		/****************image display and save************/
		apCopyMatrix1Duc(img_recon_c, img_in.data, img_in_size_bytes);

		//convert from unsigned to signed
		// img_in.convertTo(img_in, 1);
		cv::cvtColor(img_in, img_in, cv::COLOR_GRAY2RGB);
	    
		std::cout << "\n\nimg_out width: " << img_in.cols << "\n" <<
		"img_out height: " << img_in.rows << "\n" <<
		"img_out depth: " << img_in.depth() << "\n" <<
		"img_out channels " << img_in.channels() << "\n" <<
		"img_out type: " << img_in.type() << "\n";

		// std::cout << "\n\nimg_out width: " << img_out.cols << "\n" <<
		// "img_out height: " << img_out.rows << "\n" <<
		// "img_out size: " << img_out.size().width << "*" << img_out.size().height << "\n" <<
		// "img_out depth: " << img_out.depth() << "\n" <<
		// "img_out channels " << img_out.channels() << "\n" <<
		// "img_out type: " << img_out.type() << "\n";


		//find out where to write the output to:
        std::size_t lastDirPos = img_in_path_str.find_last_of("/");
        std::string directory = img_in_path_str.substr(0, lastDirPos);
        std::string fileName = img_in_path_str.substr(lastDirPos+1, img_in_path_str.length());
        std::string img_out_path = path_out + "/" + fileName;
		cv::imwrite(img_out_path, img_in);

		// cv::imshow("img_in", img_in);
		// // cv::imshow("img_out", img_out);
	    
	 //    cvWaitKey(0);
		// cvDestroyAllWindows();
    }



	apReleaseMatrix1Dd(img_d);
	apReleaseMatrix1Dd(largeScale_exp_d);
	apReleaseMatrix1Dd(smallScale_exp_d);
	apReleaseMatrix2Dd(smallScale_exp_2d,img_in.rows);
	apReleaseMatrix2Dd(img_2d, img_in.rows);
	apReleaseMatrix2Dd(largescale_2d, img_in.rows);
	apReleaseMatrix2Dd(smallscale_2d, img_in.rows);
	apReleaseMatrix1Duc(img_recon_c);
	largedct_cvmat.release();
	img_in.release();
	// img_out.release();
    

	return 0;
	/*******************************************************************************************************/
}