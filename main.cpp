#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <cmath>


#include "apLTV.hpp"
#include "apmyDCT.hpp"
#include "apImgProcess.hpp"
#include "apSmallScaleAdjust.hpp"

 
using namespace cv;

int main()
{

	/********************************************************************************************************
	****************************    HS **********************************************************************
    *********************************************************************************************************/
	// unsigned char* img_uc;
	double* img_d;
	double* largeScale_exp_d;
	double **img_2d;
	double *smallScale_exp_d;
	double **smallScale_exp_2d;
	// double *img_recon_d;
	// double *data5;
	unsigned char *img_recon_c;
	double **largescale_2d;
	double **smallscale_2d;
	int i;
	CvMat * smallAdjust, *img_re;
    cv::Mat largedct_cvmat;
	// IplImage* img_c = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/dataset/A_1.png", 0);
	cv::Mat img_c = cv::imread("/home/hooman/Illumination-Normalization-For-Face-Images/dataset/A_1.png" , cv::IMREAD_GRAYSCALE);
	cv::Mat img_out = img_c.clone();

	size_t img_c_size_bytes = img_c.total() * img_c.elemSize();

	if (img_c.empty())
	{
		printf("can't open the image...\n");

	}

	

	/*****image decomposition************/
	//convert char data to unsigned char data. hs: no longer needed since cv::Mat is unsigned char
	// img_uc = apCtoUC(img_c.data, img_c_size_bytes);

	//log transform, also transforms the data to double
	img_d = apLogUC2(img_c.data, img_c_size_bytes);

	//matrix transform to 2d
	img_2d = ap1DTo2Dd(img_d, img_c.rows, img_c.cols);

	printf("conduct LTV.....\n");//decompose source image to a small-scale image and a large-scale image

	//matrix transform back to 1d
	largescale_2d = apLTV(img_2d, img_c.rows, img_c.cols, 0.4, 0.1, 100);
	smallscale_2d = apGetSmallScale(img_2d, largescale_2d, img_c.rows, img_c.cols);
	//  apDataPrint2Dd(smallscale_2d,img_c.rows, img_c.cols,".\\1.txt");


	/*************image process******************/
	// normalizing illumination
	largedct_cvmat = apLargeScale_LogDCT(img_c.rows, img_c.cols, largescale_2d); // normalize large-scale image


	/************image reconstruction*************/
	//conduct exponential transform
	largeScale_exp_d = apExp1D((double*)largedct_cvmat.data, img_c_size_bytes);//large
	
	//conduct exponential transform
	smallScale_exp_2d = apExp2D(smallscale_2d, img_c.rows, img_c.cols);//small
	smallScale_exp_d = ap2DTo1Dd(smallScale_exp_2d, img_c.rows, img_c.cols);

	//hs
	// IplImage* hs_img_larg = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);
	// IplImage* hs_img_small = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);
	// auto hs_larg_1d = ap2DTo1Dd(largescale_2d, img_c.rows, img_c.cols);
	// auto hs_small_1d = ap2DTo1Dd(smallscale_2d, img_c.rows, img_c.cols);

	// auto hs_larg_c = apDtoC_strengthened(hs_larg_1d, hs_img_larg->imageSize, 100);
	// auto hs_small_c = apDtoC_strengthened(hs_small_1d, hs_img_small->imageSize, 100);

	// apCopyMatrix1Dc(hs_larg_c, hs_img_larg->imageData, hs_img_larg->imageSize);
	// apCopyMatrix1Dc(hs_small_c, hs_img_small->imageData, hs_img_small->imageSize);



	//apDataPrint2Dd(img_2d,img_c.rows, img_c.cols,".\\1.txt");

	for (i = 0; i < img_c_size_bytes; i++)
	{
		img_d[i] = largeScale_exp_d[i] * smallScale_exp_d[i];
	}

	//transform double data to char data with 100 times strengthened
	img_recon_c = apDtoC_strengthened(img_d, img_c_size_bytes, 100);


	/****************image display and save************/
	// apCopyMatrix1Dc(img_recon_c, img_out.data, img_c_size_bytes);
	apCopyMatrix1Duc(img_recon_c, img_out.data, img_c_size_bytes);
    printf("conduct display img_c.....\n");
	
	// cvNamedWindow("img_c", 1);
	// cvShowImage("img_c", img_c);
	
	// cvNamedWindow("hs_img_larg", 2);
	// cvShowImage("hs_img_larg", hs_img_larg);
	
	// cvNamedWindow("hs_img_small", 3);
	// cvShowImage("hs_img_small", hs_img_small);

	std::cout << "\n\nimg_c width: " << img_c.cols << "\n" << "img_c height: " << img_c.rows << "\n" << "img_c size: " << img_c.size().width << "*" << img_c.size().height << "\n" << "img_c depth: " << img_c.depth() << "\n" << "img_c channels " << img_c.channels() << "\n" << "img_c type: " << img_c.type() << "\n";

	std::cout << "\n\nimg_out width: " << img_out.cols << "\n" <<
	"img_out height: " << img_out.rows << "\n" <<
	"img_out size: " << img_out.size().width << "*" << img_out.size().height << "\n" <<
	"img_out depth: " << img_out.depth() << "\n" <<
	"img_out channels " << img_out.channels() << "\n" <<
	"img_out type: " << img_out.type() << "\n";

	cv::imshow("img_c", img_c);
	cv::imshow("img_out", img_out);
    
    cvWaitKey(0);
    //cv::destroyWindow("Image");
	cvDestroyAllWindows();

	apReleaseMatrix1Duc(img_c.data);
	apReleaseMatrix1Dd(img_d);
	apReleaseMatrix1Dd(largeScale_exp_d);
	apReleaseMatrix1Dd(smallScale_exp_d);
	apReleaseMatrix2Dd(smallScale_exp_2d,img_c.rows);
	apReleaseMatrix2Dd(img_2d, img_c.rows);
	apReleaseMatrix2Dd(largescale_2d, img_c.rows);
	apReleaseMatrix2Dd(smallscale_2d, img_c.rows);
	// apReleaseMatrix1Dd(img_recon_d);
	//apReleaseMatrix1Dd(data5);
	apReleaseMatrix1Duc(img_recon_c);

	img_c.release();
	img_out.release();

	//cv::cvReleaseMat(&largedct_cvmat);
    largedct_cvmat.release();

	return 0;
	/*******************************************************************************************************/
}