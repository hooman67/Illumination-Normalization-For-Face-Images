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
	unsigned char* img_uc;
	double* img_d;
	double* largeScale_exp_d;
	double **img_2d;
	double *smallScale_exp_d;
	double **smallScale_exp_2d;
	// double *img_recon_d;
	// double *data5;
	char *img_recon_c;
	double **largescale_2d;
	double **smallscale_2d;
	int i;
	CvMat * smallAdjust, *img_re;
    cv::Mat largedct_cvmat;
	IplImage* img_c = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);
	if (!img_c)
	{
		printf("can't open the image...\n");

	}

	IplImage* hs_img_larg = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);
	IplImage* hs_img_small = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);



	/*****image decomposition************/
	//convert char data to unsigned char data
	img_uc = apCtoUC(img_c->imageData, img_c->imageSize);

	//log transform, also transforms the data to double
	img_d = apLogUC2(img_uc, img_c->imageSize);

	//matrix transform to 2d
	img_2d = ap1DTo2Dd(img_d, img_c->height, img_c->width);

	printf("conduct LTV.....\n");//decompose source image to a small-scale image and a large-scale image

	//matrix transform back to 1d
	largescale_2d = apLTV(img_2d, img_c->height, img_c->width, 0.4, 0.1, 100);
	smallscale_2d = apGetSmallScale(img_2d, largescale_2d, img_c->height, img_c->width);
	//  apDataPrint2Dd(smallscale_2d,img_c->height, img_c->width,".\\1.txt");


	/*************image process******************/
	// normalizing illumination
	largedct_cvmat = apLargeScale_LogDCT(img_c->height, img_c->width, largescale_2d); // normalize large-scale image


	/************image reconstruction*************/
	//conduct exponential transform
	largeScale_exp_d = apExp1D((double*)largedct_cvmat.data, img_c->imageSize);//large
	
	//conduct exponential transform
	smallScale_exp_2d = apExp2D(smallscale_2d, img_c->height, img_c->width);//small
	smallScale_exp_d = ap2DTo1Dd(smallScale_exp_2d, img_c->height, img_c->width);

	//hs
	auto hs_larg_1d = ap2DTo1Dd(largescale_2d, img_c->height, img_c->width);
	auto hs_small_1d = ap2DTo1Dd(smallscale_2d, img_c->height, img_c->width);

	auto hs_larg_c = apDtoC_strengthened(hs_larg_1d, hs_img_larg->imageSize, 100);
	auto hs_small_c = apDtoC_strengthened(hs_small_1d, hs_img_small->imageSize, 100);

	apCopyMatrix1Dc(hs_larg_c, hs_img_larg->imageData, hs_img_larg->imageSize);
	apCopyMatrix1Dc(hs_small_c, hs_img_small->imageData, hs_img_small->imageSize);



	//apDataPrint2Dd(img_2d,img_c->height, img_c->width,".\\1.txt");

	for (i = 0; i < img_c->imageSize; i++)
	{
		img_d[i] = largeScale_exp_d[i] * smallScale_exp_d[i];
	}

	//transform double data to char data with 100 times strengthened
	img_recon_c = apDtoC_strengthened(img_d, img_c->imageSize, 100);


	/****************image display and save************/
	apCopyMatrix1Dc(img_recon_c, img_c->imageData, img_c->imageSize);
    printf("conduct display img_c.....\n");
	
	cvNamedWindow("img_c", 1);
	cvShowImage("img_c", img_c);
	
	cvNamedWindow("hs_img_larg", 2);
	cvShowImage("hs_img_larg", hs_img_larg);
	
	cvNamedWindow("hs_img_small", 3);
	cvShowImage("hs_img_small", hs_img_small);
    
    cvWaitKey(0);
    //cv::destroyWindow("Image");
	cvDestroyAllWindows();

	apReleaseMatrix1Duc(img_uc);
	apReleaseMatrix1Dd(img_d);
	apReleaseMatrix1Dd(largeScale_exp_d);
	apReleaseMatrix1Dd(smallScale_exp_d);
	apReleaseMatrix2Dd(smallScale_exp_2d,img_c->height);
	apReleaseMatrix2Dd(img_2d, img_c->height);
	apReleaseMatrix2Dd(largescale_2d, img_c->height);
	apReleaseMatrix2Dd(smallscale_2d, img_c->height);
	// apReleaseMatrix1Dd(img_recon_d);
	//apReleaseMatrix1Dd(data5);
	apReleaseMatrix1Dc(img_recon_c);

	//cv::cvReleaseMat(&largedct_cvmat);
    largedct_cvmat.release();
	cvReleaseImage(&img_c);

	return 0;
	/*******************************************************************************************************/


	/********************************************************************************************************
	test: image decomposition (one source image decompose to one small-scale image and one large-scale image)
   *********************************************************************************************************/
   // unsigned char* data1;
   // double* data2;
   // double **data3;
   // double *data4;
   // double *data5;
   // char *data6;
   // double **largescale_2d;
   // double **smallscale_2d;
   // int i;

   // //read in

   // IplImage* img = cvLoadImage(".\\1.bmp",0);
   // if( !img )  
   // {  
   //    printf("can't open the image...\n");  

   // } 

   // //convert char data to unsigned char data
   // data1=apCtoUC(img->imageData, img->imageSize);

   // //log transform, also transforms the data to double
   // data2=apLogUC2(data1,img->imageSize);

   // //matrix transform to 2d
   // data3=ap1DTo2Dd(data2,img->height,img->width);

   // printf("conduct LTV.....\n");

   // largescale_2d=apLTV(data3,img->height, img->width, 0.4, 0.1,100);
   // smallscale_2d=apGetSmallScale(data3, largescale_2d, img->height, img->width);

   // //matrix transform back to 1d
   // data4=ap2DTo1Dd(smallscale_2d,img->height, img->width);

   // //conduct exponential transform, and then increase the output 100 times for display purpose
   // // data5=apExp1D_strengthened(data4,img->imageSize);
   
   // //transform  double data to char data
   // data6=apDtoC(data5,img->imageSize);

   // apCopyMatrix1Dc(data6,img->imageData,img->imageSize);
   // printf("conduct display img.....\n");
   // cvNamedWindow( "Image", 1 );
   // cvShowImage( "Image", img );
   // cvWaitKey(0); 
   // cvDestroyWindow("Image");
   // cvSaveImage(".\\small.jpg",img,0);

   // apReleaseMatrix1Duc(data1);
   // apReleaseMatrix1Dd(data2);
   // apReleaseMatrix2Dd(data3,img->height);
   // apReleaseMatrix2Dd(largescale_2d,img->height);
   // apReleaseMatrix2Dd(smallscale_2d,img->height);
   // apReleaseMatrix1Dd(data4);
   // apReleaseMatrix1Dd(data5);
   // apReleaseMatrix1Dc(data6);

   // cvReleaseImage( &img);

/******************************************************************
test: adjust small-scale images
**********************************************************************/
//char threshold;
//IplImage *src = 0;
//IplImage *dst = 0;

//src = cvLoadImage(".\\small_s3.jpg", 0);
//dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
//dst = cvCloneImage(src);


//threshold = apFindThreshold_data(src->imageData, src->nSize);
////  threshold=apFindThreshold(src);
//apMidFilter_Thre(src, dst, threshold);

//cvNamedWindow("src", 0);
//cvNamedWindow("dst", 0);
//cvShowImage("src", src);
//cvShowImage("dst", dst);

//cvWaitKey(-1);
//cvDestroyWindow("src");
//cvDestroyWindow("dst");
//cvSaveImage(".\\smalladjust.jpg", dst, 0);
//cvReleaseImage(&src);
//cvReleaseImage(&dst);

//void smallAdjust(double **data, int row, int col)
//{
//	threshold = apFindThreshold_data(src->imageData, src->nSize);
//	apMidFilter_Thre(src, dst, threshold);
//}


/******************************************************************
test: illumination normalization on the large-scale images
**********************************************************************/
//unsigned char* data1;
//double* data2;
//double **data3;
//double *data5;
//char *data6;
//double **largescale_2d;
//int i;
//CvMat *largedct_cvmat;

//IplImage* img = cvLoadImage(".\\1.bmp", 0);
//if (!img)
//{
//	printf("can't open the image...\n");

//}

//data1 = apCtoUC(img->imageData, img->imageSize);

//data2 = apLogUC2(data1, img->imageSize);
//data3 = ap1DTo2Dd(data2, img->height, img->width);

//printf("conduct LTV.....\n");

//largescale_2d = apLTV(data3, img->height, img->width, 0.4, 0.1, 100);
////conduct logDCT
//largedct_cvmat = apLargeScale_LogDCT(img->height, img->width, largescale_2d);//largescale_2d logDCT
////exp
//data5 = apExp1D_strengthened(largedct_cvmat->data.db, img->imageSize);
//// double to char

//data6 = apDtoC(data5, img->imageSize);

//printf("conduct new img.....\n");
//apCopyMatrix1Dc(data6, img->imageData, img->imageSize);

//printf("conduct display img.....\n");
//cvNamedWindow("Image", 1);
//cvShowImage("Image", img);
//cvWaitKey(0);
//cvDestroyWindow("Image");
//cvSaveImage(".\\Snorm.jpg", img, 0);

//apReleaseMatrix1Duc(data1);
//apReleaseMatrix1Dd(data2);
//apReleaseMatrix2Dd(data3, img->height);
//apReleaseMatrix2Dd(largescale_2d, img->height);
//apReleaseMatrix1Dd(data5);
//apReleaseMatrix1Dc(data6);
//cvReleaseMat(&largedct_cvmat);
//cvReleaseImage(&img);
/*****************************************************************************************
test: image fuse of the adjusted small-scale image and the normalized large-scale image
*****************************************************************************************/
//IplImage *smallscale_2d ;
//IplImage *large ;
//IplImage *img_out;
//CvMat* Msmall;
//CvMat* Mlarge;
//CvMat* Msum;
//int i,j;
//double t;
//smallscale_2d = cvLoadImage(".\\smalladjust.jpg", 0 );
//large = cvLoadImage(".\\Snorm_s3.jpg", 0 );
//img_out=cvCreateImage(cvSize(smallscale_2d->width,smallscale_2d->height), smallscale_2d->depth, 1);
//
//Msmall = cvCreateMat(smallscale_2d->height,smallscale_2d->width,CV_32SC1);
//Mlarge = cvCreateMat(large->height,large->width,CV_32SC1);
//Msum = cvCreateMat(large->height,large->width,CV_32SC1);
//cvConvert( smallscale_2d, Msmall); 
//cvConvert( large, Mlarge); 
//cvMul(Msmall,Mlarge, Msum,1);
//
////for(i=0;i<large->height;i++)
////for(j=0;j<large->width;j++)
////{
////    t = cvmGet(Msum,i,j); // Get M(i,j) 
////    
////   // cvmSet(Msum,i,j,100+t); // Set M(i,j)
////}
////cvCrossProduct(Msmall,Mlarge, Msum);
//cvConvert( Msum, img_out);
//
//cvSaveImage(".\\sum.jpg",img_out,0);
//cvReleaseImage( &smallscale_2d);
//cvReleaseImage( &large);
//cvReleaseImage( &img_out);
//cvReleaseMat(&Msmall);  
//cvReleaseMat(&Mlarge);  
//cvReleaseMat(&Msum);  

/******************************************************************************************************************
test: whole system (image decomposation + adjust small-scale image + normalize large-scale image + image fusing)
*******************************************************************************************************************/
	// unsigned char* data1;
	// double* data2;
	// double **data3;
	// double *data4;
	// double *data5;
	// char *data6;
	// double **largescale_2d;
	// double **smallscale_2d;
	// int i;
	// CvMat * smallAdjust, *img_re;
 //    cv::Mat largedct_cvmat;
	// IplImage* img = cvLoadImage("/home/hooman/Illumination-Normalization-For-Face-Images/boy.bmp", 0);
	// if (!img)
	// {
	// 	printf("can't open the image...\n");

	// }
	// /*****image decomposition************/
	// data1 = apCtoUC(img->imageData, img->imageSize);

	// data2 = apLogUC2(data1, img->imageSize);
	// data3 = ap1DTo2Dd(data2, img->height, img->width);

	// printf("conduct LTV.....\n");//decompose source image to a small-scale image and a large-scale image

	// largescale_2d = apLTV(data3, img->height, img->width, 0.4, 0.1, 100);

	// smallscale_2d = apGetSmallScale(data3, largescale_2d, img->height, img->width);
	// //  apDataPrint2Dd(smallscale_2d,img->height, img->width,".\\1.txt");

	// /*************image process******************/
	// largedct_cvmat = apLargeScale_LogDCT(img->height, img->width, largescale_2d); // normalize large-scale image


	// /************image reconstruction*************/
	// data2 = apExp1D((double*)largedct_cvmat.data, img->imageSize);//large

	// data3 = apExp2D(smallscale_2d, img->height, img->width);//small
	// data4 = ap2DTo1Dd(data3, img->height, img->width);
	// //apDataPrint2Dd(data3,img->height, img->width,".\\1.txt");

	// for (i = 0; i < img->imageSize; i++)
	// {
	// 	data4[i] = data2[i] * data4[i];
	// }
	// data6 = apDtoC_strengthened(data4, img->imageSize);

	// /****************image display and save************/
	// apCopyMatrix1Dc(data6, img->imageData, img->imageSize);
 //    printf("conduct display img.....\n");
	// cvNamedWindow("Image", 1);
	// cvShowImage("Image", img);
 //    cvWaitKey(0);
 //    //cv::destroyWindow("Image");
	// cvDestroyAllWindows();
	// auto hsIm = cvCreateMat(img,CV_32SC1);
	// imwrite("/home/hooman/Illumination-Normalization-For-Face-Images/boyRes.jpg", img);

	// apReleaseMatrix1Duc(data1);
	// apReleaseMatrix1Dd(data2);
	// apReleaseMatrix2Dd(data3, img->height);
	// apReleaseMatrix2Dd(largescale_2d, img->height);
	// apReleaseMatrix2Dd(smallscale_2d, img->height);
	// apReleaseMatrix1Dd(data4);
	// //apReleaseMatrix1Dd(data5);
	// apReleaseMatrix1Dc(data6);

	// //cv::cvReleaseMat(&largedct_cvmat);
 //    largedct_cvmat.release();
	// cvReleaseImage(&img);

    // return 0;
}