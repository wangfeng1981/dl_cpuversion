#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
/*#include <random> it seems cuda4.0 not support c++ 11 */
/*
 * 20170625 9:44 完成：
 * 图像、卷积、池层强项传播卷积层，
 * 卷积层前向传播池层，
 * 池层前向全连接，
 * 全连接后向池层，
 * 池层后向卷积，
 * 卷积层后向池层、图像、卷积层，
 * 卷积层计算kernel weights sum，
 * 计算bias和weights的平均更新值。
 * 下一步整合卷积层和池层进入整个网络运算！！   
 * 
 * 20170626-0724 here 完成后续卷积层、池层、全连接层的前向连接，
 *                    然后完成后向传播，然后对变化求和，然后更新weights。
 * 
 * 20170709 全连接层前向传播和softmax没有问题了
 *          检查到卷积后向传播到池层 修复了bugs，后向传播基本没有问题了。
 * 20170710 修改随机数种子 这步十分必要，
 *          每次计算都要使用不同的种子进行初始化，
 *          否则一旦碰上无法收敛的情况则永远无法收敛。
 * 20170711 修改softmax bugs
 * 20170713 ReLU
 * 20170714 Dropout
 * */


/*
 * VS2008 编译cpp文件obj
 * 1.启动VS Command Prompt
 * 2.输入命令 cl /Fo -c cppfullpath.cpp
 * 3.obj文件在命令行窗口的工作目录下
 */
 
/*
 * nvcc 编译cu文件成obj
 *  * 去掉 USE_GPU_MODE前面的注释
 * nvcc -c -arch=sm_20 gpucnn.cu
 * 
 */
 
/**
分别编译obj以后，连接到一起生成exe
1.启动VS2008(VS9.0) Command Prompt
2.cd跳转到各个obj的目录下
3.运行下面命令
cl gpucnn.obj array3d.obj lodepng.obj wftools.obj wImage.obj jsoncpp.obj "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\lib\Win32\cudart.lib"
 
  * 
  * */


//#define USE_GPU_MODE 1


//输出debug数据到文件
//#define USE_DEBUG_OUTPUT 1
//#define USE_DEBUG_OUTPUT2 1

#ifdef USE_DEBUG_OUTPUT
std::string g_debug_output_filename ;
FILE* g_debug_output_fileptr = 0 ;
#endif

#ifdef USE_DEBUG_OUTPUT2
std::string g_debug_output_filename2 ;
FILE* g_debug_output_fileptr2 = 0 ;
#endif


#include "../../sharedcppcodes/array3d.h" 
#include "../../sharedcppcodes/wImage.h" 
#include "../../sharedcppcodes/wftools.h" 
#include "../../sharedcppcodes/json.h" 




#ifndef USE_GPU_MODE
void cudaMalloc( float** pPtr , int n ){
}
void cudaFree(float* ptr){
}
const int cudaMemcpyHostToDevice = 0;
const int cudaMemcpyDeviceToHost = 1;
void cudaMemcpy( float* targetPtr , float* sourcePtr , int n , int mode ) {
}
#endif

#define SAFE_RELEASE(p) if(p){delete p;p=0;}

//高斯随机数 这个随机数很重要
//初始化bias和weights的时候
//必须使用这个高斯随机数否则不容易收敛
float gaussrandvalue( float muval , float stdval ){
	static float V1, V2, S;
	static int phase = 0;
	float X;
	if(phase == 0) {
		do {
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
        }
        while(S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);
	} else {
	    X = V2 * sqrt(-2 * log(S) / S);
	}
	phase = 1 - phase;
	return muval + X * stdval ;
}
extern int writeArrayToFile(const char* prefix1,const char* prefix2,int index1,
						const char* mid ,
						int index2,
						const char* tail,float* array,int nx,int ny,int nz) ;
						
extern void writeDebugFloatArray( const char* name , int index , float* farr , int nx,int ny,int nz,int nk );
extern void writeDebugLine( const char* name , int index  ) ;
extern void writeDebugLineFloat( const char* name , float val  ) ;
extern std::string currentDateTimeString( std::string ext);
extern void getTopNIndexFromArray( float* array , int arraysize , int n , int* iarr , float* varr ) ;




class GFloatArray{
public:
	inline GFloatArray(int nfloat,bool fillRand):
		m_hostMemory(0),
		m_devMemory(0),
		m_nfloat(nfloat),
		m_nbytes(0)
	{
		this->m_nbytes=sizeof(float)*m_nfloat;
		m_hostMemory=new float[m_nfloat];
		float sc = sqrtf( 1.0f / m_nfloat ) ;
		if( fillRand ){
			for(int i = 0 ; i<m_nfloat ; ++ i ){
				m_hostMemory[i] = gaussrandvalue(0.0f,sc) ;
			}
		}else{
			memset(m_hostMemory , 0 , m_nbytes ) ;
		}
		cudaMalloc(&m_devMemory,m_nbytes);
		this->copyHost2Device() ;
	} ;

	inline ~GFloatArray(){
		m_nfloat=0;m_nbytes=0;
		delete[] m_hostMemory;m_hostMemory=0;
		cudaFree(m_devMemory) ;m_devMemory = 0 ;
	} ;
	
	inline void copyFromArrayHost( GFloatArray* fromArray ){
		assert( this->m_nfloat == fromArray->getNFloat() ) ;
		for(int i = 0 ; i<this->m_nfloat ; ++ i ){
			this->getHostMemory()[i] = fromArray->getHostMemory()[i] ;
		}
		#ifdef USE_GPU_MODE
		this->copyHost2Device() ;
		#endif
	} ;

	inline float* getHostMemory(){ return m_hostMemory ; } ;
	inline float* getDevMemory(){return m_devMemory; } ;
	inline int getNFloat(){return m_nfloat;} ;
	inline int getNBytes(){return m_nbytes;} ;
	inline void copyHost2Device(){
		cudaMemcpy( m_devMemory , m_hostMemory , m_nbytes , cudaMemcpyHostToDevice ) ;
	} ;
	inline void copyDeviceToHost(){
		cudaMemcpy( m_hostMemory , m_devMemory , m_nbytes , cudaMemcpyDeviceToHost ) ;
	} ;
private:
	float* m_hostMemory ;
	float* m_devMemory ;
	int m_nfloat ;
	int m_nbytes ;
} ;
//=======================================================================================
//=======================================================================================
//=======================================================================================

class GLabeledData{
public:
	inline GLabeledData():m_dataPtr(0),m_label(0){} ;
	GFloatArray* m_dataPtr ;
	int          m_label ;
	int          m_id ;
} ;




//=======================================================================================
//=======================================================================================
//=======================================================================================
enum GLayerType { GLayerTypeNone , GLayerTypeFull , GLayerTypeConv ,GLayerTypePool , GLayerTypeFastRCNNOutput };

class GLayer{
public:
	GLayerType m_type ;
	std::string m_layerName ;
	bool        m_fixWeightsAndBias ;
	inline GLayerType getType() { return m_type ;} ;
	virtual ~GLayer() ;
	virtual Json::Value toJsonNode() ;
} ;
GLayer::~GLayer() {
}
Json::Value GLayer::toJsonNode() {
	Json::Value node ;
	return node ;
}

class GLayerFull : public GLayer {
public:
	GLayerFull(int insize , int outsize ) ;
	GLayerFull(Json::Value& jsonNode ) ;
	~GLayerFull() ;
	Json::Value toJsonNode() ;
	
	GFloatArray* m_actiArray ;
	GFloatArray* m_biasAndWeights ;
	GFloatArray* m_biasAndWeightsChangesSum ;
	GFloatArray* m_errorArray ;
	GFloatArray* m_lastBiasAndWeightsChanges ;
	GFloatArray* m_dropoutMaskArray ;// 0.0 for dropout , 1.0 for activ
	int m_insize , m_outsize ;
	bool m_useDropoutMask ;
	
	void shuffleDropoutMaskArray() ;
	void setAllMaskOne() ;
} ;
GLayerFull::~GLayerFull(){
	SAFE_RELEASE(this->m_actiArray) ;
	SAFE_RELEASE(this->m_biasAndWeights) ;
	SAFE_RELEASE(this->m_biasAndWeightsChangesSum) ;
	SAFE_RELEASE(this->m_errorArray) ;
	SAFE_RELEASE(this->m_lastBiasAndWeightsChanges) ;
	SAFE_RELEASE(this->m_dropoutMaskArray) ;
}

Json::Value GLayerFull::toJsonNode() {
	Json::Value node ;
	
	node["layer-name"] = this->m_layerName ;
	node["layer-type"] = this->m_type ;
	node["inpux-x-size"] = m_insize ;
	node["inpux-y-size"] = 1 ;
	node["inpux-z-size"] = 1 ;
	node["output-x-size"] = m_outsize ;
	node["output-y-size"] = 1 ;
	node["output-z-size"] = 1 ;
	node["wb-x-size"] = m_outsize ;
	node["wb-y-size"] = m_insize + 1 ;
	node["wb-z-size"] = 1 ;
	node["wb-k-size"] = 1 ;
	node["drop-out"] = this->m_useDropoutMask ;
	node["fix-weights-bias"] = this->m_fixWeightsAndBias ;
	
	int nwb = this->m_biasAndWeights->getNFloat() ;
	#ifdef USE_GPU_MODE 
	this->m_biasAndWeights->copyDeviceToHost() ;
	#endif
	for(int i = 0 ; i<nwb ; ++ i ){
		node["wb"][i] = this->m_biasAndWeights->getHostMemory()[i] ;
	}
	
	return node ;
}

GLayerFull::GLayerFull(int insize , int outsize  ){
	this->m_insize = insize ;
	this->m_outsize = outsize ;
	this->m_type = GLayerTypeFull ;
	this->m_actiArray = new GFloatArray(outsize,false) ;
	this->m_dropoutMaskArray = new GFloatArray(outsize,false) ;
	this->m_useDropoutMask = false ;
	this->m_fixWeightsAndBias = false ;
	
	int bwsize = outsize + outsize * insize ;
	this->m_biasAndWeights = new GFloatArray(bwsize,true) ;
	this->m_biasAndWeightsChangesSum = new GFloatArray(bwsize,false) ;
	this->m_errorArray = new GFloatArray(outsize,false) ;
	this->m_lastBiasAndWeightsChanges = new GFloatArray(bwsize,false) ;
	
	for(int i = 0 ; i<this->m_dropoutMaskArray->getNFloat() ; ++ i ){
		this->m_dropoutMaskArray->getHostMemory()[i] = 1.0f ;
	}
	#ifdef USE_GPU_MODE
	this->m_dropoutMaskArray->copyHost2Device() ;
	#endif
}

GLayerFull::GLayerFull(Json::Value& jsonNode ){
	m_type = (GLayerType)jsonNode["layer-type"].asInt() ;
	assert( m_type == GLayerTypeFull ) ;
	m_insize = jsonNode["input-x-size"].asInt() ;
	m_outsize = jsonNode["output-x-size"].asInt() ;

	this->m_actiArray = new GFloatArray(m_outsize,false) ;
	this->m_dropoutMaskArray = new GFloatArray(m_outsize,false) ;
	
	if( jsonNode.isMember("drop-out") ){
		this->m_fixWeightsAndBias = jsonNode["drop-out"].asBool() ;
	}else{
		this->m_fixWeightsAndBias = false ;
	}
	
	if( jsonNode.isMember("fix-weights-bias") ){
		this->m_fixWeightsAndBias = jsonNode["fix-weights-bias"].asBool() ;
	}else{
		this->m_fixWeightsAndBias = false ;
	}
	
	int bwsize = m_outsize + m_outsize * m_insize ;
	
	//卷积核数组
	if( jsonNode.isMember("wb") && jsonNode["wb"].size() > 0 ){
		this->m_biasAndWeights = new GFloatArray( bwsize , false ) ;
		int nwb = (int)jsonNode["wb"].size() ;
		for(int i = 0 ; i<nwb ; ++ i ){
			m_biasAndWeights->getHostMemory()[i] = jsonNode["wb"][i].asFloat() ;
		}
		#ifdef USE_GPU_MODE
		m_biasAndWeights->copyHost2Device() ;
		#endif
	}else{
		this->m_biasAndWeights = new GFloatArray(bwsize,true) ;
	}
	this->m_biasAndWeightsChangesSum = new GFloatArray(bwsize,false) ;
	this->m_errorArray = new GFloatArray(m_outsize,false) ;
	this->m_lastBiasAndWeightsChanges = new GFloatArray(bwsize,false) ;
	
	for(int i = 0 ; i<this->m_dropoutMaskArray->getNFloat() ; ++ i ){
		this->m_dropoutMaskArray->getHostMemory()[i] = 1.0f ;
	}
	#ifdef USE_GPU_MODE
	this->m_dropoutMaskArray->copyHost2Device() ;
	#endif
	
	
	
}

void GLayerFull::shuffleDropoutMaskArray() {
	if( this->m_useDropoutMask ){
		int size = this->m_dropoutMaskArray->getNFloat() ;
		int halfsize = this->m_dropoutMaskArray->getNFloat()/2 ;
		int nMask0 = 0 ;
		for(int i = 0 ; i<size ; ++ i ){
			this->m_dropoutMaskArray->getHostMemory()[i] = 1.0f ;
		}
		while( nMask0 < halfsize ){
			int rIndex = rand()%size ;
			if( this->m_dropoutMaskArray->getHostMemory()[rIndex] > 0.5f ){
				this->m_dropoutMaskArray->getHostMemory()[rIndex] = 0.0f ;
				++ nMask0 ;
			}
		}
		#ifdef USE_GPU_MODE
		this->m_dropoutMaskArray->copyHost2Device() ;
		#endif
	}
}
void GLayerFull::setAllMaskOne() {
	if( this->m_useDropoutMask ){
		int size = this->m_dropoutMaskArray->getNFloat() ;
		for(int i = 0 ; i<size ; ++ i ){
			this->m_dropoutMaskArray->getHostMemory()[i] = 1.0f ;
		}
		#ifdef USE_GPU_MODE
		this->m_dropoutMaskArray->copyHost2Device() ;
		#endif
	}
}

//=======================================================================================
//=======================================================================================
//=======================================================================================
class GLayerFastRCNNOutput : public GLayer {
public:
	GLayerFastRCNNOutput(int insize , int outsizeNoBack ) ;
	~GLayerFastRCNNOutput() ;
	GFloatArray* m_actiArraySoftmax ; // K+1 class
	GFloatArray* m_biasAndWeightsSoftmax ;
	GFloatArray* m_biasAndWeightsChangesSumSoftmax ;
	GFloatArray* m_lastBiasAndWeightsChangesSoftmax ;
	
	GFloatArray* m_actiArrayLoc ;// K class * 4
	GFloatArray* m_biasAndWeightsLoc ;
	GFloatArray* m_biasAndWeightsChangesSumLoc ;
	GFloatArray* m_lastBiasAndWeightsChangesLoc ;
	
	
	int m_insize , m_outsizeNoBack , m_outsizeWithBack ;
	
} ;
GLayerFastRCNNOutput::~GLayerFastRCNNOutput(){
	SAFE_RELEASE(this->m_actiArraySoftmax) ;
	SAFE_RELEASE(this->m_biasAndWeightsSoftmax) ;
	SAFE_RELEASE(this->m_biasAndWeightsChangesSumSoftmax) ;
	SAFE_RELEASE(this->m_lastBiasAndWeightsChangesSoftmax) ;

	SAFE_RELEASE(this->m_actiArrayLoc) ;
	SAFE_RELEASE(this->m_biasAndWeightsLoc) ;
	SAFE_RELEASE(this->m_biasAndWeightsChangesSumLoc) ;
	SAFE_RELEASE(this->m_lastBiasAndWeightsChangesLoc ) ;
}
GLayerFastRCNNOutput::GLayerFastRCNNOutput(int insize , int outsizeNoBack  ){
	this->m_insize = insize ;
	this->m_outsizeNoBack = outsizeNoBack;
	this->m_outsizeWithBack = this->m_outsizeNoBack + 1 ;
	this->m_type = GLayerTypeFastRCNNOutput ;
	
	
	int bwsize = m_outsizeWithBack + m_outsizeWithBack * insize ;
	this->m_actiArraySoftmax = new GFloatArray(m_outsizeWithBack,false) ;
	this->m_biasAndWeightsSoftmax = new GFloatArray(bwsize,true) ;
	this->m_biasAndWeightsChangesSumSoftmax = new GFloatArray(bwsize,false) ;
	this->m_lastBiasAndWeightsChangesSoftmax = new GFloatArray(bwsize,false) ;
	
	int bwsizeloc = m_outsizeNoBack * 4 + m_outsizeNoBack * 4 * insize ;
	this->m_actiArrayLoc = new GFloatArray(m_outsizeNoBack*4,false) ;
	this->m_biasAndWeightsLoc = new GFloatArray(bwsizeloc,true) ;
	this->m_biasAndWeightsChangesSumLoc = new GFloatArray(bwsizeloc,false) ;
	this->m_lastBiasAndWeightsChangesLoc = new GFloatArray(bwsizeloc,false) ;
}

void cpu_fastrcnnOutput_forwardFromFull( 
	float* fastSoftmaxArray , 
	float* fastLocArray , 
	int fastLocSize , 
	int fastSoftmaxSize  ,
	float* fastSoftmaxBiasAndWeights ,
	float* fastLocBiasAndWeights , 
	float* prevFullActiArray , int prevFullActiSize ){
	
	assert( fastSoftmaxSize*4 - 4 == fastLocSize  ) ;
		
	//iout==0 is background softmax
	for(int iout = 0 ; iout < fastSoftmaxSize ; ++ iout ){
		float sum1 = fastSoftmaxBiasAndWeights[iout] ;
		int iw = fastSoftmaxSize + iout ;
		for( int iprev = 0 ; iprev < prevFullActiSize ; ++ iprev ){
			sum1 += fastSoftmaxBiasAndWeights[iw] * prevFullActiArray[iprev] ;
			iw += fastSoftmaxSize ;
		}
		fastSoftmaxArray[iout] = max(0.0f , sum1) ;//ReLU
	}
	
	//loc
	for(int iloc = 0  ; iloc < fastLocSize ; ++ iloc  ){
		float sum1 = fastLocBiasAndWeights[iloc] ;
		int iw = fastLocSize + iloc ;
		for( int iprev = 0 ; iprev < prevFullActiSize ; ++ iprev ){
			sum1 += fastLocBiasAndWeights[iw] * prevFullActiArray[iprev] ;
			iw += fastLocSize ;
		}
		fastLocArray[iloc] = max(0.0f , sum1) ;//ReLU
	}
		
}

void cpu_fastrcnn_backwardErrorFromLabelAndTrueRect( ){
	//做不完了，直接看faster rcnn把 20170720
}






//=======================================================================================
//====     FASTER R CNN      =================================
//=======================================================================================
class GLayerRPN : public GLayer {
public :
	int m_inputXSize , m_inputYSize , m_inputZSize ;
	int m_outputXSize , m_outputYSize , m_outputZSize ;
	int m_windowXSize , m_windowYSize , m_windowZSize ;
	
	
} ;

void cpu_rpn_forwardFromImage( 
	float* prevActiArray , 
	int prevActiXSize , 
	int prevActiYSize , 
	int prevActiZSize , 
	float* rpnAnchorKernelWeightsAndBias , //9 anchor matrix , no bias really?
	int windowSize , 
	float* outputRPNActiArray
)
{
	/*
	int nfPerAnchorColumn = windowSize * windowSize * prevActiZSize + 1; 
	int nfPerAnchorKernel = nfPerAnchorColumn * 6 ;
	for(int ix = 0 ; ix < prevActiXSize ; ++ ix ){//输入x从0开始
		if( ix + windowSize >= prevActiXSize ) break ;
		for(int iy = 0 ; iy < prevActiYSize ; ++ iy ){//输入y从0开始
			if( iy + windowSize >= prevActiYSize ) break ;
			for(int ia = 0 ; ia < 9 ; ++ ia ){//9个Anchor逐一计算
				float tsix[6] ;// object, not object, x,y,w,h ;
				float* anchorWeightsBiasZero = rpnAnchorKernelWeightsAndBias[nfPerAnchorKernel*ia] ;
				for(int iout = 0 ; iout < 6 ; ++ iout ){
					float sum1 = anchorWeightsBiasZero[iout] ;
					for(int iwx = 0 ; iwx < windowSize ; ++ iwx ){
						for(int iwy = 0 ; iwy < windowSize ; ++ iwy ){
							here !!
						}
					}
				}
			}
			
			
			// 9 anchor
			
		}
	}*/
	
}


















//=======================================================================================
//=======================================================================================
//=======================================================================================

//卷积层
class GLayerConv : public GLayer 
{
public:
	GLayerConv(int inx,int iny,int inz,int knx,int kny,int nk) ;
	GLayerConv(Json::Value& jsonNode ) ;
	~GLayerConv() ;
	Json::Value toJsonNode() ;
	
	int m_kernelCount ;
	int m_kernelPixelCountPerBand ;
	int m_inBandCount ;
	int m_kXsize , m_kYsize ;
	int m_ioXsize , m_ioYsize , m_ioPixelCountPerBand ;
	int m_biasStartIndex ;
	
	GFloatArray* m_actiArray ;
	GFloatArray* m_reluArray ;//激励值大于0 =1.0f 否则=0.0f 这个是否需要有待确定
	GFloatArray* m_errorArray ;
	GFloatArray* m_kernelWeightsBiasArray ;
	GFloatArray* m_kernelWeightsBiasChangeSumArray ;
	GFloatArray* m_kernelWeightsBiasLastChangeArray ;
	
} ;
//图像的像素的组成格式如下 像素顺序按照先行后列在波段 
GLayerConv::GLayerConv(int inx,int iny,int inz,int knx,int kny,int nk)
{
	m_type = GLayerTypeConv ;
	m_ioXsize = inx ; //输入输出影像的x大小
	m_ioYsize = iny ; //输入输出影像的y大小
	m_ioPixelCountPerBand = m_ioXsize * m_ioYsize ;//输入输出像素个数 m_ioPixelCount
	this->m_fixWeightsAndBias = false ;
	
	m_kXsize = knx ;  //卷积x大小必须奇数
	m_kYsize = kny ;  //卷积y大小必须奇数
	m_kernelPixelCountPerBand = m_kXsize * m_kYsize ;//卷积像素个数 m_kernelPixelCount
	
	m_kernelCount = nk ;//卷积核心数量
	m_inBandCount = inz ; //输入波段数 m_inPixelBandCount
	
	int nfloatk = m_kernelPixelCountPerBand * m_inBandCount * m_kernelCount + m_kernelCount ;
	
	//卷积核数组
	m_kernelWeightsBiasArray = new GFloatArray( nfloatk , true) ;
	m_kernelWeightsBiasChangeSumArray = new GFloatArray( nfloatk , false) ;
	m_kernelWeightsBiasLastChangeArray = new GFloatArray( nfloatk , false) ;
	
	
	//激励值数组
	int nfloatA = m_ioPixelCountPerBand * m_kernelCount ;
	m_actiArray  = new GFloatArray( nfloatA ,false ) ;
	m_errorArray = new GFloatArray( nfloatA ,false ) ;
	m_reluArray  = new GFloatArray( nfloatA ,false ) ;

}
GLayerConv::GLayerConv(Json::Value& jsonNode ) {
	m_type = (GLayerType) jsonNode["layer-type"].asInt() ;
	assert( m_type == GLayerTypeConv ) ;

	if( jsonNode.isMember("fix-weights-bias") ){
		this->m_fixWeightsAndBias = jsonNode["fix-weights-bias"].asBool() ;
	}else{
		this->m_fixWeightsAndBias = false ;
	}
	
	m_ioXsize = jsonNode["input-x-size"].asInt() ; ; //输入输出影像的x大小
	m_ioYsize = jsonNode["input-y-size"].asInt() ; ; //输入输出影像的y大小
	m_ioPixelCountPerBand = m_ioXsize * m_ioYsize ;//输入输出像素个数 m_ioPixelCount
	
	m_kXsize = jsonNode["wb-x-size"].asInt() ; ;  //卷积x大小必须奇数
	m_kYsize = jsonNode["wb-y-size"].asInt() ; ;  //卷积y大小必须奇数
	m_kernelPixelCountPerBand = m_kXsize * m_kYsize ;//卷积像素个数 m_kernelPixelCount
	
	m_kernelCount = jsonNode["wb-k-size"].asInt() ; ;//卷积核心数量
	m_inBandCount = jsonNode["input-z-size"].asInt() ; ; //输入波段数 m_inPixelBandCount
	
	int nfloatk = m_kernelPixelCountPerBand * m_inBandCount * m_kernelCount + m_kernelCount ;
	
	//卷积核数组
	if( jsonNode.isMember("wb") && jsonNode["wb"].size() > 0 ){
		m_kernelWeightsBiasArray = new GFloatArray( nfloatk , false ) ;
		int nwb = (int)jsonNode["wb"].size() ;
		for(int i = 0 ; i<nwb ; ++ i ){
			m_kernelWeightsBiasArray->getHostMemory()[i] = jsonNode["wb"][i].asFloat() ;
		}
		#ifdef USE_GPU_MODE
		m_kernelWeightsBiasArray->copyHost2Device() ;
		#endif
	}else{
		m_kernelWeightsBiasArray = new GFloatArray( nfloatk , true) ;
	}
	m_kernelWeightsBiasChangeSumArray = new GFloatArray( nfloatk , false) ;
	m_kernelWeightsBiasLastChangeArray = new GFloatArray( nfloatk , false) ;
	
	
	//激励值数组
	int nfloatA = m_ioPixelCountPerBand * m_kernelCount ;
	m_actiArray  = new GFloatArray( nfloatA ,false ) ;
	m_errorArray = new GFloatArray( nfloatA ,false ) ;
	m_reluArray  = new GFloatArray( nfloatA ,false ) ;
}
GLayerConv::~GLayerConv(){
	SAFE_RELEASE(m_kernelWeightsBiasArray) ;
	SAFE_RELEASE(m_kernelWeightsBiasChangeSumArray) ;
	SAFE_RELEASE(m_kernelWeightsBiasLastChangeArray) ;
	
	SAFE_RELEASE(m_actiArray) ;
	SAFE_RELEASE(m_errorArray) ;
	SAFE_RELEASE(m_reluArray) ;
}


Json::Value GLayerConv::toJsonNode() {
	Json::Value node ;
	
	node["layer-name"] = this->m_layerName ;
	node["layer-type"] = this->m_type ;
	node["inpux-x-size"] = this->m_ioXsize ;
	node["inpux-y-size"] = this->m_ioYsize ;
	node["inpux-z-size"] = this->m_inBandCount ;
	node["output-x-size"] = this->m_ioXsize ;
	node["output-y-size"] = this->m_ioYsize ;
	node["output-z-size"] = this->m_kernelCount ;
	node["wb-x-size"] = m_kXsize ;
	node["wb-y-size"] = m_kYsize ;
	node["wb-z-size"] = m_inBandCount ;
	node["wb-k-size"] = m_kernelCount ;
	node["fix-weights-bias"] = this->m_fixWeightsAndBias ;
	
	int nwb = this->m_kernelWeightsBiasArray->getNFloat() ;
	#ifdef USE_GPU_MODE 
	this->m_kernelWeightsBiasArray->copyDeviceToHost() ;
	#endif
	for(int i = 0 ; i<nwb ; ++ i ){
		node["wb"][i] = this->m_kernelWeightsBiasArray->getHostMemory()[i] ;
	}
	
	return node ;
}



////////////////////////////////////////////////////////////////////////////////////

//池层
class GLayerPool : public GLayer 
{
public:
	GLayerPool(int inx,int iny,int inz) ;
	~GLayerPool() ;
	GLayerPool(Json::Value& jsonNode ) ;
	Json::Value toJsonNode() ;
	
	int inXSize , inYSize , bandCount ;
	int outXSize , outYSize ;
	
	GFloatArray* m_actiArray ;
	GFloatArray* m_convIsMaxArray ;//Conv激励值于2x2最大 =1.0f 否则=0.0f
	GFloatArray* m_errorArray ;
} ;
//图像的像素的组成格式如下 像素顺序按照先行后列在波段 
GLayerPool::GLayerPool(int inx,int iny,int inz)
{
	m_type = GLayerTypePool ;
	inXSize = inx ; //输入输出影像的x大小
	inYSize = iny ; //输入输出影像的y大小
	bandCount = inz ;
	outXSize = inXSize/2 ;
	outYSize = inYSize/2 ;
	this->m_fixWeightsAndBias = false ;
	
	int nfc = inXSize * inYSize * bandCount ;
	int nfp = outXSize * outYSize * bandCount  ;
	
	//激励值数组
	m_actiArray  = new GFloatArray( nfp ,false ) ;
	m_errorArray = new GFloatArray( nfp ,false ) ;
	m_convIsMaxArray  = new GFloatArray( nfc ,false ) ;

}
GLayerPool::GLayerPool(Json::Value& jsonNode )
{
	m_type = (GLayerType) jsonNode["layer-type"].asInt() ;
	assert( m_type == GLayerTypePool ) ;

	inXSize = jsonNode["input-x-size"].asInt() ; //输入输出影像的x大小
	inYSize = jsonNode["input-y-size"].asInt() ; //输入输出影像的y大小
	bandCount = jsonNode["input-z-size"].asInt()  ;
	outXSize = inXSize/2 ;
	outYSize = inYSize/2 ;
	
	if( jsonNode.isMember("fix-weights-bias") ){
		this->m_fixWeightsAndBias = jsonNode["fix-weights-bias"].asBool() ;
	}else{
		this->m_fixWeightsAndBias = false ;
	}
	
	int nfc = inXSize * inYSize * bandCount ;
	int nfp = outXSize * outYSize * bandCount  ;
	
	//激励值数组
	m_actiArray  = new GFloatArray( nfp ,false ) ;
	m_errorArray = new GFloatArray( nfp ,false ) ;
	m_convIsMaxArray  = new GFloatArray( nfc ,false ) ;

}
GLayerPool::~GLayerPool(){
	SAFE_RELEASE(m_actiArray) ;
	SAFE_RELEASE(m_errorArray) ;
	SAFE_RELEASE(m_convIsMaxArray) ;
}

Json::Value GLayerPool::toJsonNode() {
	Json::Value node ;
	
	node["layer-name"] = this->m_layerName ;
	node["layer-type"] = this->m_type ;
	node["inpux-x-size"] = this->inXSize ;
	node["inpux-y-size"] = this->inYSize ;
	node["inpux-z-size"] = this->bandCount ;
	node["output-x-size"] = this->outXSize ;
	node["output-y-size"] = this->outYSize ;
	node["output-z-size"] = this->bandCount ;
	node["wb-x-size"] = 0 ;
	node["wb-y-size"] = 0 ;
	node["wb-z-size"] = 0 ;
	node["wb-k-size"] = 0 ;
	node["fix-weights-bias"] = this->m_fixWeightsAndBias ;
	
	return node ;
}


////////////////////////////////////////////////////////////////////////////////////

extern void unpoolingFromPoolToConv( GLayerPool* fromPoolLayer , float* toArray  ) ;
extern void unconvFromConvToImage( GLayerConv* fromConvLayer , float* toArray  ) ;




/////////////////////////////////////////////////////////////////////////////////////



//二维数据（卷积层、输入图像、池层）前向传播到卷积层
void cpu_conv_forwardFromImage(
	float* fromActi ,     /* 输入激励值 */
	int fromXSize ,       /* 输入输出影像的X方向像素个数 */
	int fromYSize ,       /* 输入输出影像的Y方向像素个数 */
	int fromZSize ,       /* 输入影像波段数量            */
	float* kernelWeightsBias ,/* 卷积核weights bias*/
	int biasStartIndex ,  /* bias 开始索引值 */
	float* outToActi ,    /* 输出激励值 */
	float* outToRelu ,    /* 输出值是否满足Relu 激励值大于0为1.0f，否则为0.0f */
	int kSize ,  /* 卷积核长宽必须一致且为奇数，比如5x5 那么kSize=5*/
	int hkSize , /* 半宽取正hkSize=kSize/2=2 */
	int kCount , /* 卷积核数量，与结果影像的波段数一致 */
	int outPixelCountPerBand ,/* = fromXSize*fromYSize ,*/
	int outSize        /* = outPixelCountPerBand*kCount  , */
	)
{
	for(int it = 0 ; it < outSize ; ++ it ){
		int cik = it / outPixelCountPerBand ;//第几个kernel
		int cipx = it % outPixelCountPerBand ;//应用了第cik个kernel的输出影像一维像素索引值
		int cix = cipx % fromXSize ;//输出影像x坐标
		int ciy = cipx / fromXSize ;//输出影像y坐标
		int kernelNFloat = kSize*kSize*fromZSize ;//每个kernel的浮点数数量
		float sum1 = kernelWeightsBias[biasStartIndex+cik] ;
		int iw = cik * kernelNFloat  ;// index of weight in the kernel.
		for(int iband = 0 ; iband < fromZSize ; ++ iband ){
			int ia1 = iband * outPixelCountPerBand ;
			for(int iy = -hkSize ; iy<= hkSize ; ++ iy ){
				int tiy = ciy + iy ;
				bool iyinside = true ;
				if( tiy < 0 || tiy >= fromYSize ) iyinside=false ;  //bugfix
				int ia0 = tiy * fromXSize ;
				for( int ix = -hkSize ; ix <= hkSize ; ++ ix ){
					int tix = cix + ix ;
					bool ixinside = true ;
					if( tix < 0 || tix >= fromXSize ) ixinside=false ;  //bugfix
					
					if( iyinside && ixinside ){ //bugfix
						int ia = ia1 + ia0 + tix ;
						float w = kernelWeightsBias[iw] ; 
						float a = fromActi[ia] ;
						sum1 += w*a ;
					}
					++iw ;
				}
			}
		}
		if( sum1 > 0.0f ){
			outToRelu[it] = 1.0f ;
			outToActi[it] = sum1 ;
		}else{
			outToRelu[it] = 0.0f ;
			outToActi[it] = 0.0f ;
		}
	}
}

#ifdef USE_GPU_MODE 
//图像数据的组织形式为 先行后列最后是波段
//卷积核数据的组织形式为 先行后列再波段最后核
/*
 * 比如 (row,col,band) == (y,x,z)
 * 0,0,0  0,1,0  0,2,0  0,3,0 ...
 * 1,0,0  1,1,0  1,2,0  1,3,0 ...
 * 
 * 0,0,1  0,1,1  0,2,1 ...
 * 1,0,1  1,1,1  1,2,1 ...
 * 
 * */
//图像层，池层或卷积层前向传播至卷积层
__global__
void gpu_conv_forwardFromImage(
	float* fromActi ,     /* 输入激励值 */
	int fromXSize ,       /* 输入输出影像的X方向像素个数 */
	int fromYSize ,       /* 输入输出影像的Y方向像素个数 */
	int fromZSize ,       /* 输入影像波段数量            */
	float* kernelWeightsBias ,/* 卷积核weights bias*/
	int biasStartIndex ,  /* bias 开始索引值 */
	float* outToActi ,    /* 输出激励值 */
	float* outToRelu ,    /* 输出值是否满足Relu 激励值大于0为1.0f，否则为0.0f */
	int kSize ,  /* 卷积核长宽必须一致且为奇数，比如5x5 那么kSize=5*/
	int hkSize , /* 半宽取正hkSize=kSize/2=2 */
	int kCount , /* 卷积核数量，与结果影像的波段数一致 */
	int outPixelCount ,/* = fromXSize*fromYSize ,*/
	int outSize        /* = outPixelCount*kCount  , */
	)
{
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < outSize ){
		int cik = it / outPixelCount ;//第几个kernel
		int cipx = it % outPixelCount ;//应用了第cik个kernel的输出影像一维像素索引值
		int cix = cipx % fromXSize ;//输出影像x坐标
		int ciy = cipx / fromXSize ;//输出影像y坐标
		int kernelNFloat = kSize*kSize*fromZSize ;//每个kernel的浮点数数量
		float sum1 = kernelWeightsBias[biasStartIndex+cik] ;
		int iw = cik * kernelNFloat  ;// index of weight in the kernel.
		for(int iband = 0 ; iband < fromZSize ; ++ iband ){
			int ia1 = iband * outPixelCount ;
			for(int iy = -hkSize ; iy<= hkSize ; ++ iy ){
				int tiy = ciy + iy ;
				//if( tiy < 0 || tiy >= fromYSize ) continue ;
				bool iyinside = true ;
				if( tiy < 0 || tiy >= fromYSize ) iyinside=false ;  //bugfix
				
				int ia0 = tiy * fromXSize ;
				for( int ix = -hkSize ; ix <= hkSize ; ++ ix ){
					int tix = cix + ix ;
					//if( tix < 0 || tix >= fromXSize ) continue ; this line cause iw can not ++ , then sum1 is wrong value for edege pixels.
					bool ixinside = true ;
					if( tix < 0 || tix >= fromXSize ) ixinside=false ;  //bugfix
					
					if( iyinside && ixinside ){ //bugfix
						int ia = ia1 + ia0 + tix ;
						float w = kernelWeightsBias[iw] ; 
						float a = fromActi[ia] ;
						sum1 += w*a ;
					}
					++iw ;
				}
			}
		}
		if( sum1 > 0.0f ){
			outToRelu[it] = 1.0f ;
			outToActi[it] = sum1 ;
		}else{
			outToRelu[it] = 0.0f ;
			outToActi[it] = 0.0f ;
		}
	}
}
#endif

//卷积层前向传播至池层
void cpu_pool_forwardFromImage(
	float* fromActi ,     /* 输入激励值 */
	int fromXSize ,       /* 输入输出影像的X方向像素个数 */
	int fromYSize ,       /* 输入输出影像的Y方向像素个数 */
	int fromZSize ,       /* 输入影像波段数量            */
	int fromPixelCount ,  /* 输入影像像素数量 fromXSize*fromYSize */
	float* outToActi ,    /* 输出激励值 2x2缩小重采样 */
	float* outToIsmax ,    /* 对应fromActi 该值为2x2最大为1.0f，否则为0.0f */
	int outXSize ,   /* 输出X大小 */
	int outYSize ,   /* 输出Y大小 */
	int outPixelCountPerBand ,/* = outXSize*outYSize ,*/
	int outSize        /* = outPixelCountPerBand*fromZSize  , */
	)
{
	for(int it = 0 ; it < outSize ; ++ it ) {
		int iband = it / outPixelCountPerBand ;//输出影像第几个波段
		int cipx =  it % outPixelCountPerBand ;//输出影像一维像素索引值
		int cix = cipx % outXSize ;//输出影像x坐标
		int ciy = cipx / outYSize ;//输出影像y坐标
		
		
		int ia0 = iband*fromPixelCount + ciy*2*fromXSize + cix*2 ;
		int ia1 = ia0 + 1 ;
		int ia2 = ia0 + fromXSize ;
		int ia3 = ia2 + 1 ;
		
		outToIsmax[ia0] = 0.0f ;
		outToIsmax[ia1] = 0.0f ;
		outToIsmax[ia2] = 0.0f ;
		outToIsmax[ia3] = 0.0f ;
		
		float amax = fromActi[ia0] ;// bug fix 20170704
		int imax = ia0 ;
		
		if( fromActi[ia1] > amax ){
			imax = ia1 ;
			amax = fromActi[ia1] ;
		}
		
		if( fromActi[ia2] > amax ){
			imax = ia2 ;
			amax = fromActi[ia2] ;
		}
		
		if( fromActi[ia3] > amax ){
			imax = ia3 ;
			amax = fromActi[ia3] ;
		}
		
		outToIsmax[imax] = 1.0f ;
		outToActi[it] = amax ;
	}
}

#ifdef USE_GPU_MODE 
//卷积层前向传播至池层
__global__
void gpu_pool_forwardFromImage(
	float* fromActi ,     /* 输入激励值 */
	int fromXSize ,       /* 输入输出影像的X方向像素个数 */
	int fromYSize ,       /* 输入输出影像的Y方向像素个数 */
	int fromZSize ,       /* 输入影像波段数量            */
	int fromPixelCount ,  /* 输入影像像素数量 fromXSize*fromYSize */
	float* outToActi ,    /* 输出激励值 2x2缩小重采样 */
	float* outToIsmax ,    /* 对应fromActi 该值为2x2最大为1.0f，否则为0.0f */
	int outXSize ,   /* 输出X大小 */
	int outYSize ,   /* 输出Y大小 */
	int outPixelCount ,/* = outXSize*outYSize ,*/
	int outSize        /* = outPixelCount*fromZSize  , */
	)
{
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < outSize ){
		int iband = it / outPixelCount ;//输出影像第几个波段
		int cipx =  it % outPixelCount ;//输出影像一维像素索引值
		int cix = cipx % outXSize ;//输出影像x坐标
		int ciy = cipx / outYSize ;//输出影像y坐标
		
		
		int ia0 = iband*fromPixelCount + ciy*2*fromXSize + cix*2 ;
		int ia1 = ia0 + 1 ;
		int ia2 = ia0 + fromXSize ;
		int ia3 = ia2 + 1 ;
		
		outToIsmax[ia0] = 0.0f ;
		outToIsmax[ia1] = 0.0f ;
		outToIsmax[ia2] = 0.0f ;
		outToIsmax[ia3] = 0.0f ;
		
		float amax = fromActi[ia0] ;// bug fix 20170704
		int imax = ia0 ;
		
		if( fromActi[ia1] > amax ){
			imax = ia1 ;
			amax = fromActi[ia1] ;
		}
		
		if( fromActi[ia2] > amax ){
			imax = ia2 ;
			amax = fromActi[ia2] ;
		}
		
		if( fromActi[ia3] > amax ){
			imax = ia3 ;
			amax = fromActi[ia3] ;
		}
		
		outToIsmax[imax] = 1.0f ;
		outToActi[it] = amax ;
	}
}
#endif


#ifdef USE_GPU_MODE 
//全连接后向传播至池层
//原理与全连接层一致，使用 void gpu_full_backwardErrorFromNextLayer(...)
	
#endif

// 池层后向传播至卷积层  
void cpu_conv_backwardErrorFromPoolLayer(
	float* poolError ,
	float* poolConvActiIsMax ,
	int poolPxPerBandCount , /* 池层波段像素个数 */
	int poolXSize , 
	int outFSize , /* conv层全部acti值数量 */
	int outPxPerBand , /* 卷积层每个波段像素数量 */
	int outXSize , /* 卷积层行像素个数 */
	float* outToError
	) {
	
	for(int it = 0 ; it < outFSize ; ++ it ){
		if( poolConvActiIsMax[it] > 0.5f ){
			//激励值为2x2最大时 = 1.0f 否则=0.0f 取0.5作为判断依据
			int iband = it / outPxPerBand ;
			int ipx1d = it % outPxPerBand ; //像素一维index
			int ix = ipx1d % outXSize ;
			int iy = ipx1d / outXSize ;
			int ipx = ix/2 ;
			int ipy = iy/2 ;
			int ipool = iband * poolPxPerBandCount + ipy * poolXSize + ipx ;
			outToError[it] = poolError[ipool] ;
		}else{
			outToError[it] = 0.0f ;
		}
	}
}
#ifdef USE_GPU_MODE 
// 池层后向传播至卷积层 
__global__
void gpu_conv_backwardErrorFromPoolLayer(
	float* poolError ,
	float* poolConvActiIsMax ,
	int poolPxPerBandCount , /* 池层波段像素个数 */
	int poolXSize , 
	int outFSize , /* conv层全部acti值数量 */
	int outPxPerBand , /* 卷积层每个波段像素数量 */
	int outXSize , /* 卷积层行像素个数 */
	float* outToError
	) {
	
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < outFSize ){
		if( poolConvActiIsMax[it] > 0.5f ){
			//激励值为2x2最大时 = 1.0f 否则=0.0f 取0.5作为判断依据
			int iband = it / outPxPerBand ;
			int ipx1d = it % outPxPerBand ; //像素一维index
			int ix = ipx1d % outXSize ;
			int iy = ipx1d / outXSize ;
			int ipx = ix/2 ;
			int ipy = iy/2 ;
			int ipool = iband * poolPxPerBandCount + ipy * poolXSize + ipx ;
			outToError[it] = poolError[ipool] ;
		}else{
			outToError[it] = 0.0f ;
		}
	}
}
#endif

// 卷积层后向传播误差至池层、卷积层、图像层
void cpu_image_backwardErrorFromConvLayer(
	float* convError ,
	float* convActi ,
	float* convKernelWeightsBiasArray ,
	int convActiXSize ,
	int convActiYSize , 
	int convNFloatPerBand , 
	int nFloatPerKernel , 
	int kernelXHalfSize , 
	int kernelXSize ,
	int kernelCount ,
	int outFSize ,/* 池层总激励值个数 */
	int outPxPerBandCount , /* 池层波段像素个数 */
	int outXSize , 
	float* outToError
	) {
	
	int nfloatPerKernelBand = kernelXSize*kernelXSize ;
	for(int it = 0 ; it < outFSize ; ++ it ){
		//完成卷积层后向传递到池层
		int iPoolBand = it / outPxPerBandCount ;
		int iPool1d = it % outPxPerBandCount ;
		int ipx = iPool1d % outXSize ;
		int ipy = iPool1d / outXSize ;
		
		float sum1 = 0.0f ;
		for( int kx = -kernelXHalfSize ; kx <= kernelXHalfSize ; ++ kx ){
			int iconvx = kx + ipx ;
			int ikernelx = kernelXHalfSize - kx ;
			if( iconvx < 0 || iconvx >= convActiXSize ) continue ;
			for(int ky = -kernelXHalfSize ; ky <= kernelXHalfSize ; ++ ky ){
				int iconvy = ky + ipy ;
				int ikernely = kernelXHalfSize - ky ;
				if( iconvy < 0 || iconvy >= convActiYSize ) continue ;
				for(int ikernel = 0 ; ikernel < kernelCount ; ++ ikernel ){
					int ierror = convNFloatPerBand * ikernel + iconvy * convActiXSize + iconvx ;
					float error1 = convError[ierror] ;
					//int iweight = iPoolBand * nFloatPerKernel + kernelXSize * ikernely + ikernelx ; 
					int iweight = ikernel * nFloatPerKernel + iPoolBand * nfloatPerKernelBand + kernelXSize * ikernely + ikernelx ; //bugfixed 20170709
					float weight1 = convKernelWeightsBiasArray[iweight] ;
					sum1 += error1 * weight1 ;
				}
			}
		}
		outToError[it] = sum1 ;
	}
}
#ifdef USE_GPU_MODE 
// 卷积层后向传播误差至池层、卷积层、图像层
__global__
void gpu_image_backwardErrorFromConvLayer(
	float* convError ,
	float* convActi ,
	float* convKernelWeightsBiasArray ,
	int convActiXSize ,
	int convActiYSize , 
	int convNFloatPerBand , 
	int nFloatPerKernel , 
	int kernelXHalfSize , 
	int kernelXSize ,
	int kernelCount ,
	int outFSize ,/* 池层总激励值个数 */
	int outPxPerBandCount , /* 池层波段像素个数 */
	int outXSize , 
	float* outToError
	) {
	
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < outFSize ){
		int nfloatPerKernelBand = kernelXSize*kernelXSize ;
		//完成卷积层后向传递到池层
		int iPoolBand = it / outPxPerBandCount ;
		int iPool1d = it % outPxPerBandCount ;
		int ipx = iPool1d % outXSize ;
		int ipy = iPool1d / outXSize ;
		
		float sum1 = 0.0f ;
		for( int kx = -kernelXHalfSize ; kx <= kernelXHalfSize ; ++ kx ){
			int iconvx = kx + ipx ;
			int ikernelx = kernelXHalfSize - kx ;
			if( iconvx < 0 || iconvx >= convActiXSize ) continue ;
			for(int ky = -kernelXHalfSize ; ky <= kernelXHalfSize ; ++ ky ){
				int iconvy = ky + ipy ;
				int ikernely = kernelXHalfSize - ky ;
				if( iconvy < 0 || iconvy >= convActiYSize ) continue ;
				for(int ikernel = 0 ; ikernel < kernelCount ; ++ ikernel ){
					int ierror = convNFloatPerBand * ikernel + iconvy * convActiXSize + iconvx ;
					float error1 = convError[ierror] ;
					//int iweight = iPoolBand * nFloatPerKernel + kernelXSize * ikernely + ikernelx ; 
					int iweight = ikernel * nFloatPerKernel + iPoolBand * nfloatPerKernelBand + kernelXSize * ikernely + ikernelx ; //bugfixed 20170709
					float weight1 = convKernelWeightsBiasArray[iweight] ;
					sum1 += error1 * weight1 ;
				}
			}
		}
		outToError[it] = sum1 ;
	}
}
#endif



//卷积层计算卷积核weights的变化值并求和
void cpu_conv_computeAndSumKernelWeightsChanges(
	float* prevActi ,
	int    prevXSize ,
	int    prevYSize ,
	float* currError ,
	int    nFloatPerActiBand , 
	int    nkernel , 
	int    kXSize ,
	int    kXHalfSize , 
	int    nFloatPerKernel , 
	int    nFloatPerKernelBand , 
	float* outWeightsBiasChangeSum ,
	int    wbSize ,
	int    biasStartIndex
	) {
	
	for(int it = 0 ; it<wbSize ; ++ it ){
		if( it >= biasStartIndex ){
			int ikernel = it - biasStartIndex ;
			int iout0 = nFloatPerActiBand * ikernel ;
			for(int iout = 0 ; iout < nFloatPerActiBand ; ++ iout ){
				outWeightsBiasChangeSum[it] += currError[iout0+iout] ;//20170713 night
			}
		}else{
			int ikernel = it / nFloatPerKernel ;
			int ik1d = it % nFloatPerKernel ;
			int iband = ik1d / nFloatPerKernelBand ;
			int ik2d  = ik1d % nFloatPerKernelBand ;
			int kerx = ik2d % kXSize ;
			int kery = ik2d / kXSize ;
			
			int iprev0 = nFloatPerActiBand * iband ;
			int iout0 =  nFloatPerActiBand * ikernel ;
			for(int imgx = 0 ; imgx < prevXSize ; ++ imgx ){
				int ox = imgx + kXHalfSize - kerx ;
				if( ox < 0 || ox >= prevXSize ) continue ;
				for(int imgy = 0 ; imgy < prevYSize ; ++ imgy ) {
					int oy = imgy + kXHalfSize - kery ;
					if( oy < 0 || oy >= prevYSize ) continue ;
					int iprev = iprev0 + imgy * prevXSize + imgx ;
					int iout = iout0 + oy * prevXSize + ox ;
					outWeightsBiasChangeSum[it] += prevActi[iprev]*currError[iout] ;
				}
			}
		}
	}
}

#ifdef USE_GPU_MODE
__global__
void gpu_conv_computeAndSumKernelWeightsChanges(
	float* prevActi ,
	int    prevXSize ,
	int    prevYSize ,
	float* currError ,
	int    nFloatPerActiBand , 
	int    nkernel , 
	int    kXSize ,
	int    kXHalfSize , 
	int    nFloatPerKernel , 
	int    nFloatPerKernelBand , 
	float* outWeightsBiasChangeSum ,
	int    wbSize , 
	int    biasStartIndex
	) {
//卷积层计算卷积核weights的变化值并求和
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < wbSize ){
		if( it >= biasStartIndex ){
			int ikernel = it - biasStartIndex ;
			int iout0 = nFloatPerActiBand * ikernel ;
			for(int iout = 0 ; iout < nFloatPerActiBand ; ++ iout ){
				outWeightsBiasChangeSum[it] += currError[iout0+iout] ;//20170713 night
			}
		}else{
			int ikernel = it / nFloatPerKernel ;
			int ik1d = it % nFloatPerKernel ;
			int iband = ik1d / nFloatPerKernelBand ;
			int ik2d  = ik1d % nFloatPerKernelBand ;
			int kerx = ik2d % kXSize ;
			int kery = ik2d / kXSize ;
			
			int iprev0 = nFloatPerActiBand * iband ;
			int iout0 =  nFloatPerActiBand * ikernel ;
			for(int imgx = 0 ; imgx < prevXSize ; ++ imgx ){
				int ox = imgx + kXHalfSize - kerx ;
				if( ox < 0 || ox >= prevXSize ) continue ;
				for(int imgy = 0 ; imgy < prevYSize ; ++ imgy ) {
					int oy = imgy + kXHalfSize - kery ;
					if( oy < 0 || oy >= prevYSize ) continue ;
					int iprev = iprev0 + imgy * prevXSize + imgx ;
					int iout = iout0 + oy * prevXSize + ox ;
					outWeightsBiasChangeSum[it] += prevActi[iprev]*currError[iout] ;
				}
			}
		}
	}
}
#endif




//=======================================================================================
//=======================================================================================
//=======================================================================================

//计算sigmoid  统一使用ReLU，不再使用sigmoid
float sigmoid( float val ){
	return 1.0f/(1.0f+exp(-val))  ;
}



//计算前向传播激活值 ReLU
void cpu_full_forwardFromPrevLayer(
	float* fromActi ,
	int fromSize ,
	float* toBiasAndWeights ,
	float* outToActi ,
	int toSize , 
	float* dropoutMask
	){
	for(int iout = 0 ; iout < toSize ; ++ iout ){
		float sum1 = toBiasAndWeights[iout] ;
		int iw = toSize + iout ;
		for( int iprev = 0 ; iprev < fromSize ; ++ iprev ){
			sum1 += toBiasAndWeights[iw] * fromActi[iprev] ;
			iw += toSize ;
		}
		//outToActi[iout] = sigmoid(sum1) ;
		if( dropoutMask == 0 ){
			outToActi[iout] = max(0.0f , sum1) ;//ReLU dropout
		}else{
			outToActi[iout] = max(0.0f , sum1) * dropoutMask[iout] ;//ReLU dropout
		}
		
	}
}
//计算输出层前向传播激活值 softmax 第一步
void cpu_output_forwardFromPrevLayer(
	float* fromActi ,
	int fromSize ,
	float* toBiasAndWeights ,
	float* outToActi ,
	int toSize
	){
	for(int iout = 0 ; iout < toSize ; ++ iout ){
		float sum1 = toBiasAndWeights[iout] ;
		int iw = toSize + iout ;
		for( int iprev = 0 ; iprev < fromSize ; ++ iprev ){
			sum1 += toBiasAndWeights[iw] * fromActi[iprev] ;
			iw += toSize ;
		}
		outToActi[iout] = exp(sum1) ;
	}
}
//全连接层前向传播
#ifdef USE_GPU_MODE
__global__
void gpu_full_forwardFromPrevLayer(float* fromActi ,
	int fromSize ,
	float* toBiasAndWeights ,
	float* outToActi ,
	int toSize , 
	float* dropoutMask
	){
		
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < toSize ){
		float sum1 = toBiasAndWeights[it] ;
		int iw = toSize + it ;
		for( int iprev = 0 ; iprev < fromSize ; ++ iprev ){
			sum1 += toBiasAndWeights[iw] * fromActi[iprev] ;
			iw += toSize ;
		}
		//outToActi[it] = 1.0f/(1.0f+exp(-sum1))  ;
		if( dropoutMask == 0 ){
			outToActi[it] = max(0.0f,sum1)  ;//ReLU  here 20170713 next check softmax and backprop , with dropout
		}else{
			outToActi[it] = max(0.0f,sum1) * dropoutMask[it] ;//ReLU  here 20170713 next check softmax and backprop , with dropout
		}
		
	}
}
//softmax 输出层计算激励值 第一步
__global__
void gpu_output_forwardFromPrevLayer(float* fromActi ,
	int fromSize ,
	float* toBiasAndWeights ,
	float* outToActi ,
	int toSize
	){
		
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < toSize ){
		float sum1 = toBiasAndWeights[it] ;
		int iw = toSize + it ;
		for( int iprev = 0 ; iprev < fromSize ; ++ iprev ){
			sum1 += toBiasAndWeights[iw] * fromActi[iprev] ;
			iw += toSize ;
		}
		outToActi[it] = exp(sum1)  ;//softmax
	}
}
#endif



//计算输出层激励值 softmax 第二步
int cpu_output_computeOutputSoftmaxValues(
	float* actiArr ,
	int size ,
	float* outPossibility
	){
	
	float softmaxsum = 0.0f ;
	for(int i = 0 ; i<size ; ++ i ){
		softmaxsum += actiArr[i] ;
	}
	if( softmaxsum < 0.000001f ){
		std::cout<<"softmaxsum almost zero:"<<softmaxsum<<std::endl ;
	}
	actiArr[0] = actiArr[0]/softmaxsum ;
	float maxposs = actiArr[0] ;
	int iguess = 0 ;
	for(int i = 1 ; i<size ; ++ i ){
		actiArr[i] = actiArr[i] / softmaxsum ;
		if( actiArr[i] > maxposs ){
			maxposs = actiArr[i] ;
			iguess = i ;
		}
	}
	*outPossibility = maxposs ;
	return iguess ;
}
#ifdef USE_GPU_MODE

//计算输出层激励值 softmax 第二步
__global__
void gpu_output_computeOutputSoftmaxValues(
	float* actiArr ,
	int size ,
	int target ,
	float* goodBadMseArr 
	){
	//only 1 thread use
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < 1 ){
		float softmaxsum = 0.0f ;
		for(int i = 0 ; i<size ; ++ i ){
			softmaxsum += actiArr[i] ;
		}
		if( fabsf(softmaxsum) < 0.000001f ){
			softmaxsum = 1.0f ;
		}
		float mse = 0.0f ;
		int guess = 0 ;
		float poss = -1.0f ;
		for(int i = 0 ; i<size ; ++ i ){
			actiArr[i] = actiArr[i] / softmaxsum ;
			if( actiArr[i] > poss ){
				guess = i ;
				poss = actiArr[i] ;
			}
		}
		mse = -log(actiArr[target]) ; //log-likelihood cost function
		goodBadMseArr[2] = goodBadMseArr[2] + mse ;
		if( guess == target ){
			goodBadMseArr[0] = goodBadMseArr[0] + 1.0f ;
		}else{
			goodBadMseArr[1] = goodBadMseArr[1] + 1.0f ;
		}
	}
}



//计算输出层激励值 softmax 第二步
__global__
void gpu_output_computeOutputSoftmaxValuesAndWeights01(
	float* actiArr ,
	int size ,
	int target , 
	float* weightsArr ,
	float* goodBadMseArr 
	){
	//only 1 thread use
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < 1 ){
		float softmaxsum = 0.0f ;
		for(int i = 0 ; i<size ; ++ i ){
			softmaxsum += actiArr[i] ;
		}
		if( fabsf(softmaxsum) < 0.000001f ){
			softmaxsum = 1.0f ;
		}
		float mse = 0.0f ;
		int guess = 0 ;
		float poss = -1.0f ;
		for(int i = 0 ; i<size ; ++ i ){
			actiArr[i] = actiArr[i] / softmaxsum ;
			if( actiArr[i] > poss ){
				guess = i ;
				poss = actiArr[i] ;
			}
		}
		mse =-log(actiArr[target])  ; //log-likelihood cost function
		goodBadMseArr[2] = goodBadMseArr[2] + mse ;
		if( guess == target ){
			goodBadMseArr[0] = goodBadMseArr[0] + 1.0f ;
		}else{
			goodBadMseArr[1] = goodBadMseArr[1] + 1.0f ;
		}
		goodBadMseArr[3] = weightsArr[0] ;
		goodBadMseArr[4] = weightsArr[1] ;
	}
}

#endif




//计算输出层的mse
float computeMse( float* actiArray , int size , int itarget ){
	float mse = 0.0f ;
	/*
	for(int i = 0 ; i<size ; ++ i ){
		float y = 0.0f ;
		if( i==itarget ) y = 1.0f ;
		float e1 = y - actiArray[i] ;
		mse += e1*e1 ;
	}
	mse = sqrtf(mse/(size-1)) ;
	 */
	mse = -log(actiArray[itarget]) ; //log-likelihood cost function  C=-lnaLy
	return mse ;
}

//全连接层后向传递误差delta pd(C)/pd(z)
void cpu_full_backwardErrorFromNextLayer(
	float* fromError ,
	float* fromWeights2d ,
	int fromSize ,
	float* toActi ,
	int toSize ,
	float* outToError , 
	float* dropoutMask 
	) {
	for(int ic = 0 ; ic < toSize ; ++ ic ){
		float sum1 = 0.0f ;
		int iw = fromSize*(ic+1) ;
		for(int ifrom = 0 ; ifrom < fromSize ; ++ ifrom ){
			sum1 += fromError[ifrom] * fromWeights2d[iw+ifrom] ;
		}
		float partialDerivativeZ = 0.0f ;//ReLU 20170713
		if( toActi[ic] > 0.0f ){
			partialDerivativeZ = 1.0f ;
		}
		if( dropoutMask==0 ){
			outToError[ic] = sum1 * partialDerivativeZ  ;
		}else{
			outToError[ic] = sum1 * partialDerivativeZ * dropoutMask[ic] ;
		}
		
	}
}
#ifdef USE_GPU_MODE
__global__
void gpu_full_backwardErrorFromNextLayer(
	float* fromError ,
	float* fromWeights2d ,
	int fromSize ,
	float* toActi ,
	int toSize ,
	float* outToError ,
	float* dropoutMask 
	) {
	
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < toSize ){
		float sum1 = 0.0f ;
		int iw = fromSize*(it+1) ;
		for(int ifrom = 0 ; ifrom < fromSize ; ++ ifrom ){
			sum1 += fromError[ifrom] * fromWeights2d[iw+ifrom] ;
		}
		float partialDerivativeZ = 0.0f ;//ReLU 20170713
		if( toActi[it] > 0.0f ){
			partialDerivativeZ = 1.0f ;
		}
		if( dropoutMask==0 ){
			outToError[it] = sum1 * partialDerivativeZ  ;
		}else{
			outToError[it] = sum1 * partialDerivativeZ * dropoutMask[it] ;
		}
		
	}
}
#endif


//输出层计算误差
void cpu_full_backwardErrorFromLabel(
		int target ,
		float* toActi ,
		float* outToError ,
		int toSize
	)
{//softmax 是输出层激励函数，那么误差函数Cost Function使用负指数函数 Log Likelihood function
	for(int it = 0 ; it < toSize ; ++ it ){
		float y = 0.0f ;
		if( it==target ) y = 1.0f ;
		outToError[it] = toActi[it] - y  ;// activation - target book right!
	}
}
#ifdef USE_GPU_MODE
__global__
void gpu_full_backwardErrorFromLabel(
		int target ,
		float* toActi ,
		float* outToError ,
		int toSize
	)
{//softmax 是输出层激励函数，那么误差函数Cost Function使用负指数函数 Log Likelihood function
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < toSize ){
		float y = 0.0f ;
		if( it==target ) y = 1.0f ;
		outToError[it] = toActi[it] - y  ;
	}
}
#endif



//全连接层计算b和w的变化值并求和
void cpu_full_computeAndSumBiasAndWeightsChanges(
	float* prevActi ,
	int prevSize ,
	float* currError ,
	int currSize ,
	float* outBiasAndWeightsChangeSum ,
	int bwSize
	) {

	for(int ibw = 0 ; ibw < bwSize ; ++ ibw ){
		if( ibw < currSize ){
			outBiasAndWeightsChangeSum[ibw] += currError[ibw] ;
		}else{
			int ibw2 = ibw - currSize ;
			int iprev = ibw2 / currSize ;
			int icurr = ibw2 % currSize ;
			outBiasAndWeightsChangeSum[ibw] += prevActi[iprev] * currError[icurr];
		}
	}
}
#ifdef USE_GPU_MODE
__global__
void gpu_full_computeAndSumBiasAndWeightsChanges(
	float* prevActi ,
	int prevSize ,
	float* currError ,
	int currSize ,
	float* outBiasAndWeightsChangeSum ,
	int bwSize
	) {
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < currSize ){
		outBiasAndWeightsChangeSum[it] += currError[it] ;
	}else if( it<bwSize ){
		int ibw2 = it - currSize ;
		int iprev = ibw2 / currSize ;
		int icurr = ibw2 % currSize ;
		outBiasAndWeightsChangeSum[it] += prevActi[iprev] * currError[icurr];
	}
}
#endif


//使用b、w变化和求batch的变化平均值
void cpu_updateBiasAndWeights(
	float studyRate ,
	float momentum  ,
	int nsample     ,
	float* changesSum ,
	int size      ,
	float* lastChangesSum ,
	float* outValues
	) 
{
	float spn = studyRate / nsample ;
	for(int it = 0 ; it < size ; ++ it ){
		float temp = changesSum[it] * spn ;
		changesSum[it] = 0.0f ;
		float temp1 = lastChangesSum[it] * momentum - temp ;
		outValues[it] = outValues[it] + temp1 ;
		lastChangesSum[it] = -temp ;
	}
}

#ifdef USE_GPU_MODE
__global__
void gpu_updateBiasAndWeights(
	float studyRate ,
	float momentum  ,
	int   nsample   ,
	float* changesSum ,
	int    size ,
	float* lastChangesSum ,
	float* outValues 
	) 
{
	int it = blockIdx.x * blockDim.x + threadIdx.x ;
	if( it < size ){
		float spn = studyRate / nsample ;
		float temp = changesSum[it] * spn ;
		changesSum[it] = 0.0f ;
		float temp1 = lastChangesSum[it] * momentum - temp ;
		outValues[it] = outValues[it] + temp1 ;
		lastChangesSum[it] = -temp ;
	}
}
#endif


//=======================================================================================
//=======================================================================================
//=======================================================================================

class GDataProvider {
public:
	virtual int getDataCount() ;
	virtual void shuffle() ;
	virtual GLabeledData* getDataAt(int i) ;
	virtual int getDataDims( int* nx,int* ny,int* nz,int* nk) ;
} ;
int GDataProvider::getDataCount(){
	return 0 ;
}
void GDataProvider::shuffle(){

}
GLabeledData* GDataProvider::getDataAt(int i ){
	return 0 ;
}
int GDataProvider::getDataDims(int* nx,int* ny,int* nz,int* nk){
	if( nx != 0 ) *nx = 0 ;
	if( ny != 0 ) *ny = 0 ;
	if( nz != 0 ) *nz = 0 ;
	if( nk != 0 ) *nk = 0 ;
	return 0 ;
}


class GDataProviderDemo: public GDataProvider {
public:
	GDataProviderDemo() ;
	virtual int getDataCount() ;
	virtual void shuffle() ;
	virtual GLabeledData* getDataAt(int i) ;

	GLabeledData templabeledData ;
	float data[500] ;
	int label[100] ;
} ;
GDataProviderDemo::GDataProviderDemo(){
	int nclass = 5 ;
	for(int i = 0 ; i<100 ; ++ i ){
		label[i] = rand() % nclass ;
		data[i*5+0] = rand()*1.0f/RAND_MAX * 0.9f + label[i] ;
		data[i*5+1] = rand()*1.0f/RAND_MAX * 0.9f + label[i] ;
		data[i*5+2] = rand()*1.0f/RAND_MAX * 0.9f + label[i] ;
		data[i*5+3] = rand()*1.0f/RAND_MAX * 0.9f + label[i] ;
		data[i*5+4] = rand()*1.0f/RAND_MAX * 0.9f + label[i] ;
	}
}
int GDataProviderDemo::getDataCount(){
	return 100 ;
}
void GDataProviderDemo::shuffle(){
	for(int i = 0 ; i<100 ; ++ i ){
		int newpos = rand() % 100 ;
		float t0 = data[newpos*5+0] ;
		float t1 = data[newpos*5+1] ;
		float t2 = data[newpos*5+2] ;
		float t3 = data[newpos*5+3] ;
		float t4 = data[newpos*5+4] ;
		int tlabel = label[newpos] ;

		label[newpos] = label[i] ;
		data[newpos*5+0] = data[i*5+0] ;
		data[newpos*5+1] = data[i*5+1] ;
		data[newpos*5+2] = data[i*5+2] ;
		data[newpos*5+3] = data[i*5+3] ;
		data[newpos*5+4] = data[i*5+4] ;

		label[i] = tlabel ;
		data[i*5+0] = t0 ;
		data[i*5+1] = t1 ;
		data[i*5+2] = t2 ;
		data[i*5+3] = t3 ;
		data[i*5+4] = t4 ;
	}
}

GLabeledData* GDataProviderDemo::getDataAt(int i ){
	if( templabeledData.m_dataPtr==0 ){
		templabeledData.m_dataPtr = new GFloatArray(5,false) ;
	}
	float* harr = templabeledData.m_dataPtr->getHostMemory() ;
	harr[0] = data[i*5+0] ;
	harr[1] = data[i*5+1] ;
	harr[2] = data[i*5+2] ;
	harr[3] = data[i*5+3] ;
	harr[4] = data[i*5+4] ;
	templabeledData.m_label = label[i] ;
	return &templabeledData ;
}


//DataProvider 20
class ImgProvider : public GDataProvider {
public:
	ImgProvider(std::string filepath , float scale ) ;
	~ImgProvider() ;
	
	virtual int getDataCount() ;
	virtual void shuffle() ;
	virtual GLabeledData* getDataAt(int i) ;
	virtual int getDataDims(int*nx,int*ny,int*nz,int*nk) ;
	
	std::vector<GLabeledData*> dataVector ;
	//std::default_random_engine engine ;
	
	std::vector<std::string> labelNameVector ;
	int m_nx,m_ny,m_nz,m_nk ;
	
} ;

ImgProvider::ImgProvider(std::string filepath , float scale ){
	
	std::vector<std::string> prefixVector , tailVector  ;
	std::vector<int> fromVector , toVector ;

	
	Json::Value root ;
	std::ifstream file(filepath);
	file >> root;
	file.close() ;
	
	Json::Value samples = root["samples"] ;
	int n = (int) samples.size() ;
	int fileid = 0 ;
	for(int i = 0 ; i<n ; ++ i ){
		Json::Value s = samples[i] ;
		labelNameVector.push_back( s["label"].asString() ) ;
		int fromi = s["from"].asInt() ;
		int toi = s["to"].asInt() ;
		std::string prefix1 = s["prefix"].asString() ;
		std::string tail1 = s["tail"].asString() ;
		for(int ii = fromi ; ii < toi ; ++ ii ){
			std::stringstream ss ;
			ss<<ii ;
			std::string iistr ;
			ss>>iistr ;
			std::string filepath = prefix1 + iistr + tail1 ;
			
			wImage timg( filepath.c_str() ) ;
			int imgsize1d = timg.getCols()*timg.getRows() ;
			
			this->m_nx = timg.getCols() ;
			this->m_ny = timg.getRows() ;
			this->m_nz = 3 ;
			this->m_nk = 1 ;
			
			GLabeledData* pData = new GLabeledData() ;
			pData->m_label = i ;
			pData->m_id = fileid ++ ;
			pData->m_dataPtr = new GFloatArray(imgsize1d*3,false) ;
			float* pHost = pData->m_dataPtr->getHostMemory() ;
			for(int ipx = 0 ; ipx<imgsize1d ; ++ ipx ){
				int t0 , t1 , t2 ;
				timg.getRGB1d( ipx , t0 , t1 , t2 ) ;
				pHost[ipx] = t0*scale ; // 先行列后波段
				pHost[imgsize1d+ipx] = t1*scale ;
				pHost[imgsize1d*2+ipx] = t2*scale ;
			}
			#ifdef USE_GPU_MODE
			pData->m_dataPtr->copyHost2Device() ;
			#endif
			dataVector.push_back(pData) ;
		}
	}
}
ImgProvider::~ImgProvider() {
	int n = (int)dataVector.size() ;
	for(int i = 0 ; i<n ; ++ i ){
		SAFE_RELEASE( dataVector[i] ) ;
	}
}

int ImgProvider::getDataCount() {
	return (int) dataVector.size() ;
}

void ImgProvider::shuffle() {
	int ns = (int)this->dataVector.size() ;
	for(int i = 0 ; i<ns ; ++ i ){
		int newi = rand()%ns ;
		GLabeledData* tempPtr = dataVector[newi] ;
		dataVector[newi] = dataVector[i] ;
		dataVector[i] = tempPtr ;
	}
}

GLabeledData* ImgProvider::getDataAt(int i) {
	return dataVector[i] ;
}

int ImgProvider::getDataDims(int* nx,int* ny,int* nz,int* nk){
	if( nx != 0 ) *nx = m_nx ;
	if( ny != 0 ) *ny = m_ny ;
	if( nz != 0 ) *nz = m_nz ;
	if( nk != 0 ) *nk = m_nk ;
	return m_nx*m_ny*m_nz*m_nk ;
}





//=======================================================================================
//=======================================================================================
//=======================================================================================


class GConvNetwork{
public:
	~GConvNetwork() ;
	std::vector<GLayer*> m_layerPtrVector ;
	void run( GDataProvider* provider ,
		const int batchSize ,const int repoCount ,
		const float studyRate ,const float momentum ,
		const float finishMse ,const int savingSecs ) ;
	int guess( float* dataArray , int dataSize , float* possibility ) ;
	//void loadParams(std::string paramsfile) ;
	int m_batchSize ;
	int m_numRepos ;
	float m_studyRate ;
	float m_momentum ;
	float m_finishMse ;
	int   m_nthreadPerBlock ;
	int   m_saveSeconds ;
	float m_dataScale ;
	
	void saveToFile( const char* filepath , int fileid ) ;
	void loadFromFile( const char* filepath ) ;
	void visualizeFromActivationValue( int sampleid , int ilayer , int pixelIndex , float pixelValue ) ;
	void visualizeFromActivationValueArray(const char* prefix,  int sampleid , int ilayer , int* pixelIndexArr , float* pixelValueArr , int arrSize ) ;
	
	//write
	
} ;
GConvNetwork::~GConvNetwork() {
	for( std::vector<GLayer*>::iterator it = m_layerPtrVector.begin() ; 
			it<m_layerPtrVector.end() ; ++  it ){
		GLayer* ptr = *it ;
		SAFE_RELEASE(ptr) ;
		*it = 0 ;
	}
}

void GConvNetwork::run(GDataProvider* provider ,
	const int batchSize ,const int repoCount ,
	const float studyRate ,const float momentum ,
	const float finishMse ,const int savingSecs ) {

	int dataCount = provider->getDataCount() ;
	int layerCount = (int)m_layerPtrVector.size() ;
	if( dataCount == 0 ){
		std::cout<<"***Error*** dataCount is zero. out."<<std::endl ;
		return ;
	}
	if( layerCount==0 ){
		std::cout<<"***Error*** layerCount is zero. out."<<std::endl;
		return ;
	}

	GFloatArray goodBadMseArray(5,false) ;
	//记录开始timestamp
	unsigned saveTimestamp = (unsigned)time(NULL) ;
	
	//for repos
	for(int irepo = 0 ; irepo<repoCount ; ++ irepo ){
		#ifndef USE_DEBUG_OUTPUT 
		provider->shuffle() ; //测试时不进行乱序排列
		#endif
		
		
		int ibat = 0 ;
		int idata = 0 ;
		float repoMse = 0.0f ;
		int repoGood = 0 ;
		int repoBad = 0 ;
		
		goodBadMseArray.getHostMemory()[0] = 0.0f ;
		goodBadMseArray.getHostMemory()[1] = 0.0f ;
		goodBadMseArray.getHostMemory()[2] = 0.0f ;
		goodBadMseArray.getHostMemory()[3] = 0.0f ;
		goodBadMseArray.getHostMemory()[4] = 0.0f ;
		goodBadMseArray.copyHost2Device() ;
		
		//while for one repo
		while( idata  < dataCount ){
			GLabeledData* pData = provider->getDataAt(idata) ;
			
			#ifdef USE_DEBUG_OUTPUT2
			fprintf(g_debug_output_fileptr2 , 
			"idata:%d label:%d id:%d p0:%8.4f p1:%8.4f\n" , 
			idata , pData->m_label , pData->m_id , 
			pData->m_dataPtr->getHostMemory()[0] , 
			pData->m_dataPtr->getHostMemory()[1] ) ;
			#endif

			
			//OK
			#ifdef USE_DEBUG_OUTPUT
			
			int inputnx,inputny,inputnz,inputnk ;
			provider->getDataDims(&inputnx,&inputny,&inputnz,&inputnk) ;
			writeDebugLine("input label" , pData->m_label ) ;
			writeDebugFloatArray( "input rgb" , pData->m_id , 
									pData->m_dataPtr->getHostMemory() , 
									inputnx,
									inputny,
									inputnz,
									inputnk
								) ;
			#endif
			
			//第一步 前向传播计算估计分类
			//1 forward 不设输入层  
			//for1
			for(int ilayer = 0 ; ilayer < layerCount ; ++ ilayer ){
				GLayer* layerPtr = m_layerPtrVector[ilayer] ;
				
				if( ilayer == 0 ){
                     //第一层直接获得输入数据
					 //ok
					 
					 if( layerPtr->getType()==GLayerTypeConv ){ 
						 //卷积层
						 GLayerConv* tlayer = (GLayerConv*)layerPtr ;
						 #ifndef USE_GPU_MODE
						 cpu_conv_forwardFromImage(	
							 pData->m_dataPtr->getHostMemory() , 
							 tlayer->m_ioXsize, 
							 tlayer->m_ioYsize , 
							 tlayer->m_inBandCount ,
							 tlayer->m_kernelWeightsBiasArray->getHostMemory()  , 
							 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
							 tlayer->m_actiArray->getHostMemory() , 
							 tlayer->m_reluArray->getHostMemory() , 
							 tlayer->m_kXsize , 
							 tlayer->m_kXsize/2 , 
							 tlayer->m_kernelCount  , 
							 tlayer->m_ioXsize*tlayer->m_ioYsize , 
							 tlayer->m_actiArray->getNFloat() ) ;
							 
							 #ifdef USE_DEBUG_OUTPUT
							 writeDebugFloatArray("conv-layer" , ilayer , 
									tlayer->m_kernelWeightsBiasArray->getHostMemory() , 
									tlayer->m_kXsize , 
									tlayer->m_kXsize , 
									tlayer->m_inBandCount , 
									tlayer->m_kernelCount
								) ;
							 writeDebugFloatArray("conv-acti" , ilayer , 
									tlayer->m_actiArray->getHostMemory() , 
									tlayer->m_ioXsize , 
									tlayer->m_ioYsize , 
									tlayer->m_kernelCount , 
									1
								) ;
							 #endif
						#else
						
						gpu_conv_forwardFromImage
						<<<(tlayer->m_actiArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
						(	pData->m_dataPtr->getDevMemory() , 
							 tlayer->m_ioXsize, 
							 tlayer->m_ioYsize , 
							 tlayer->m_inBandCount ,
							 tlayer->m_kernelWeightsBiasArray->getDevMemory()  , 
							 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
							 tlayer->m_actiArray->getDevMemory() , 
							 tlayer->m_reluArray->getDevMemory() , 
							 tlayer->m_kXsize , 
							 tlayer->m_kXsize/2 , 
							 tlayer->m_kernelCount  , 
							 tlayer->m_ioXsize*tlayer->m_ioYsize , 
							 tlayer->m_actiArray->getNFloat() ) ;
						//cudaDeviceSynchronize();
						#endif
					 }else if( layerPtr->getType()==GLayerTypeFull ){
						 //全连接层  
						 GLayerFull* tlayer = (GLayerFull*)layerPtr ;
						 #ifndef USE_GPU_MODE
						 cpu_full_forwardFromPrevLayer(
											pData->m_dataPtr->getHostMemory() ,
											pData->m_dataPtr->getNFloat() ,
											tlayer->m_biasAndWeights->getHostMemory() ,
											tlayer->m_actiArray->getHostMemory() ,
											tlayer->m_actiArray->getNFloat() ,
											tlayer->m_dropoutMaskArray->getHostMemory()
											) ;
							 #ifdef USE_DEBUG_OUTPUT
							 writeDebugFloatArray("full-layer " , ilayer , 
									tlayer->m_biasAndWeights->getHostMemory() , 
									tlayer->m_outsize , 
									tlayer->m_insize+1 , 
									1 ,  
									1
								) ;
							 writeDebugFloatArray("full-acti " , ilayer , 
									tlayer->m_actiArray->getHostMemory() , 
									tlayer->m_outsize , 
									1 , 
									1 , 
									1
								) ;
							 #endif				

						#else
						gpu_full_forwardFromPrevLayer
						<<<(tlayer->m_actiArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>(
											pData->m_dataPtr->getDevMemory() ,
											pData->m_dataPtr->getNFloat() ,
											tlayer->m_biasAndWeights->getDevMemory() ,
											tlayer->m_actiArray->getDevMemory() ,
											tlayer->m_actiArray->getNFloat() ,
											tlayer->m_dropoutMaskArray->getDevMemory()
											) ;
						//cudaDeviceSynchronize();
						#endif
					 }else{
						 std::cout<<"bad layer 0"<<std::endl ;
						 exit(1) ;
					 }
					
					//ok
				}else{
					
					// 20170626-0724   后续考虑卷积层、池层、全连接层
					
					//后面网络层使用前一层的激励值作为当前输入
					GLayer* prevLayer = m_layerPtrVector[ilayer-1] ;
					
					if( layerPtr->getType() == GLayerTypeConv 
						&& prevLayer->getType()==GLayerTypePool ){
						//卷积层计算
						GLayerPool* player = (GLayerPool*)prevLayer ;
						GLayerConv* tlayer = (GLayerConv*)layerPtr ;
						#ifndef USE_GPU_MODE
						cpu_conv_forwardFromImage(
							player->m_actiArray->getHostMemory() , 
							 tlayer->m_ioXsize, 
							 tlayer->m_ioYsize , 
							 tlayer->m_inBandCount ,
							 tlayer->m_kernelWeightsBiasArray->getHostMemory()  , 
							 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
							 tlayer->m_actiArray->getHostMemory() , 
							 tlayer->m_reluArray->getHostMemory() , 
							 tlayer->m_kXsize , 
							 tlayer->m_kXsize/2 , 
							 tlayer->m_kernelCount  , 
							 tlayer->m_ioXsize*tlayer->m_ioYsize , 
							 tlayer->m_actiArray->getNFloat() 
						) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("conv-layer" , ilayer , 
								tlayer->m_kernelWeightsBiasArray->getHostMemory() , 
								tlayer->m_kXsize , 
								tlayer->m_kXsize , 
								tlayer->m_inBandCount ,  
								tlayer->m_kernelCount
							) ;
							writeDebugFloatArray("conv-acti" , ilayer , 
								tlayer->m_actiArray->getHostMemory() , 
								tlayer->m_ioXsize , 
								tlayer->m_ioYsize , 
								tlayer->m_kernelCount , 
								1
							) ;
							#endif	
						#else
						gpu_conv_forwardFromImage
						<<<(tlayer->m_actiArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
						(	player->m_actiArray->getDevMemory() , 
							 tlayer->m_ioXsize, 
							 tlayer->m_ioYsize , 
							 tlayer->m_inBandCount ,
							 tlayer->m_kernelWeightsBiasArray->getDevMemory()  , 
							 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
							 tlayer->m_actiArray->getDevMemory() , 
							 tlayer->m_reluArray->getDevMemory() , 
							 tlayer->m_kXsize , 
							 tlayer->m_kXsize/2 , 
							 tlayer->m_kernelCount  , 
							 tlayer->m_ioXsize*tlayer->m_ioYsize , 
							 tlayer->m_actiArray->getNFloat() ) ;
						//cudaDeviceSynchronize();
						#endif
						
						
					}else if( layerPtr->getType() == GLayerTypeFull 
								&& prevLayer->getType()==GLayerTypePool ) {
						//全连接层计算
						GLayerPool* player = (GLayerPool*)prevLayer ;
						GLayerFull* tlayer = (GLayerFull*)layerPtr ;
						#ifndef USE_GPU_MODE
						if( ilayer == layerCount-1 ){
							//输出层
							cpu_output_forwardFromPrevLayer(
											player->m_actiArray->getHostMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getHostMemory() ,
											tlayer->m_actiArray->getHostMemory() ,
											tlayer->m_actiArray->getNFloat()
											) ;
						}else{
							//中间层
							cpu_full_forwardFromPrevLayer(
											player->m_actiArray->getHostMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getHostMemory() ,
											tlayer->m_actiArray->getHostMemory() ,
											tlayer->m_actiArray->getNFloat() ,
											tlayer->m_dropoutMaskArray->getHostMemory()
											) ;
						}
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("full-layer" , ilayer , 
								tlayer->m_biasAndWeights->getHostMemory() , 
								tlayer->m_outsize , 
								tlayer->m_insize+1 , 
								1 ,  
								1
							) ;
							writeDebugFloatArray("full-acti" , ilayer , 
								tlayer->m_actiArray->getHostMemory() , 
								tlayer->m_outsize , 
								1 , 
								1 , 
								1
							) ;
							#endif	
						#else
						if( ilayer == layerCount-1 ){
							//输出层
							gpu_output_forwardFromPrevLayer
						<<<(tlayer->m_actiArray->getNFloat()+
							this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
							this->m_nthreadPerBlock>>>(
											player->m_actiArray->getDevMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getDevMemory() ,
											tlayer->m_actiArray->getDevMemory() ,
											tlayer->m_actiArray->getNFloat() 
											) ;
						}else{
							//中间层
							gpu_full_forwardFromPrevLayer
						<<<(tlayer->m_actiArray->getNFloat()+
							this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
							this->m_nthreadPerBlock>>>(
											player->m_actiArray->getDevMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getDevMemory() ,
											tlayer->m_actiArray->getDevMemory() ,
											tlayer->m_actiArray->getNFloat() ,
											tlayer->m_dropoutMaskArray->getDevMemory()
											) ;
						}
						
						#endif
						
						
					}else if( layerPtr->getType() == GLayerTypeFull 
								&& prevLayer->getType()==GLayerTypeFull ) {
						//全连接层计算
						GLayerFull* player = (GLayerFull*)prevLayer ;
						GLayerFull* tlayer = (GLayerFull*)layerPtr ;
						#ifndef USE_GPU_MODE
						if( ilayer == layerCount-1 ){
							//输出层
							cpu_output_forwardFromPrevLayer(
											player->m_actiArray->getHostMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getHostMemory() ,
											tlayer->m_actiArray->getHostMemory() ,
											tlayer->m_actiArray->getNFloat()
											) ;
						}else{
							//中间层
							cpu_full_forwardFromPrevLayer(
											player->m_actiArray->getHostMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getHostMemory() ,
											tlayer->m_actiArray->getHostMemory() ,
											tlayer->m_actiArray->getNFloat() ,
											tlayer->m_dropoutMaskArray->getHostMemory()
											) ;
						}
						
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("full-layer" , ilayer , 
								tlayer->m_biasAndWeights->getHostMemory() , 
								tlayer->m_outsize , 
								tlayer->m_insize+1 , 
								1 ,  
								1
							) ;
							writeDebugFloatArray("full-acti" , ilayer , 
								tlayer->m_actiArray->getHostMemory() , 
								tlayer->m_outsize , 
								1 , 
								1 , 
								1
							) ;
							#endif	
						#else
						if( ilayer == layerCount-1 ){
							//输出层
							gpu_output_forwardFromPrevLayer
						<<<(tlayer->m_actiArray->getNFloat()+
						this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
							this->m_nthreadPerBlock>>>(
											player->m_actiArray->getDevMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getDevMemory() ,
											tlayer->m_actiArray->getDevMemory() ,
											tlayer->m_actiArray->getNFloat() 
											) ;
						}else {
							//中间层
							gpu_full_forwardFromPrevLayer
						<<<(tlayer->m_actiArray->getNFloat()+
						this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
							this->m_nthreadPerBlock>>>(
											player->m_actiArray->getDevMemory() ,
											player->m_actiArray->getNFloat() ,
											tlayer->m_biasAndWeights->getDevMemory() ,
											tlayer->m_actiArray->getDevMemory() ,
											tlayer->m_actiArray->getNFloat() , 
											tlayer->m_dropoutMaskArray->getDevMemory()
											) ;
						}
						#endif
						
					}else if( layerPtr->getType() == GLayerTypePool 
								&& prevLayer->getType()==GLayerTypeConv ) {
						//池层计算
						GLayerConv* player = (GLayerConv*)prevLayer ;
						GLayerPool* tlayer = (GLayerPool*)layerPtr ;
						#ifndef USE_GPU_MODE
						cpu_pool_forwardFromImage(
							player->m_actiArray->getHostMemory() , 
							player->m_ioXsize , 
							player->m_ioYsize , 
							player->m_kernelCount , 
							player->m_ioXsize * player->m_ioYsize , 
							tlayer->m_actiArray->getHostMemory() , 
							tlayer->m_convIsMaxArray->getHostMemory() , 
							tlayer->outXSize , 
							tlayer->outYSize , 
							tlayer->outXSize * tlayer->outYSize , 
							tlayer->m_actiArray->getNFloat() ) ;
							
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("pool-layer" , ilayer , 
								tlayer->m_actiArray->getHostMemory() , 
								tlayer->outXSize , 
								tlayer->outYSize , 
								tlayer->bandCount ,  
								1
							) ;

							#endif	
						#else
						gpu_pool_forwardFromImage
						<<<(tlayer->m_actiArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
						(
							player->m_actiArray->getDevMemory() , 
							player->m_ioXsize , 
							player->m_ioYsize , 
							player->m_kernelCount , 
							player->m_ioXsize * player->m_ioYsize , 
							tlayer->m_actiArray->getDevMemory() , 
							tlayer->m_convIsMaxArray->getDevMemory() , 
							tlayer->outXSize , 
							tlayer->outYSize , 
							tlayer->outXSize * tlayer->outYSize , 
							tlayer->m_actiArray->getNFloat() 
						) ;
						//cudaDeviceSynchronize();
						#endif
						
						
					}else{
						std::cout<<"bad layer forward from"<<
							layerPtr->getType()<<" to "<<
							prevLayer->getType()
							<<std::endl ;
						exit(2) ;
					}
					
					
					
					if( ilayer == layerCount-1 ){
						GLayerFull* outlayer = (GLayerFull*)layerPtr ;
						//outlayer->m_actiArray->copyDeviceToHost() ;//need this line?
						//输出层特殊softmax处理，及计算mse等
						#ifndef USE_GPU_MODE
						float outposs = 0.0f ;
						int iguess = 
						cpu_output_computeOutputSoftmaxValues(
							outlayer->m_actiArray->getHostMemory() , 
							outlayer->m_actiArray->getNFloat() , 
							&outposs ) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("output softmax" , ilayer , 
								outlayer->m_actiArray->getHostMemory() , 
								outlayer->m_outsize , 
								1 , 
								1 , 
								1 ) ;
							writeDebugLine("iguess " , iguess) ;
							writeDebugLine("poss % " , (int)(outposs*100) ) ;
							#endif
						//将softmax值返回显存
						//outlayer->m_actiArray->copyHost2Device() ;//need this line?
						//统计样本分类正确错误和mse
						if( iguess == pData->m_label ){
							++repoGood ;
						}else{
							++repoBad ;
						}
						repoMse += computeMse( 
							outlayer->m_actiArray->getHostMemory() , 
							outlayer->m_actiArray->getNFloat() , 
							pData->m_label 
							) ;
						if( repoMse != repoMse ){
							std::cout<<"label:"<<pData->m_label<<std::endl;
							std::cout<<"size:"<<outlayer->m_actiArray->getNFloat()<<std::endl;
							for(int iout = 0 ; iout < outlayer->m_actiArray->getNFloat() ; ++ iout ){
								std::cout<<"iout"<<iout<<" : "<<outlayer->m_actiArray->getHostMemory()[iout]<<std::endl ;
							}
							std::cout<<"Bad repoMse value. NaN!"<<std::endl ;
							std::cout<<"sampleid:"<<pData->m_id<<std::endl ;
							exit(2) ;
						}
						#else 
						/*
						
						*/
						GLayer* templayer0 = m_layerPtrVector[0] ;
						if( templayer0->getType() == GLayerTypeConv ){
							GLayerConv* templayer00 = (GLayerConv*)templayer0 ;
							//获取两个weight输出查看是否有变化
							gpu_output_computeOutputSoftmaxValuesAndWeights01<<<1,1>>>(
								outlayer->m_actiArray->getDevMemory() , 
								outlayer->m_actiArray->getNFloat() ,
								pData->m_label ,
								templayer00->m_kernelWeightsBiasArray->getDevMemory() , 
								goodBadMseArray.getDevMemory() 
							) ;
						}else{
							gpu_output_computeOutputSoftmaxValues
							<<<1,1>>>(
								outlayer->m_actiArray->getDevMemory() , 
								outlayer->m_actiArray->getNFloat() ,
								pData->m_label ,
								goodBadMseArray.getDevMemory() 
							) ;
						}
						#endif
					}//endif( ilayer == layerCount-1 )
				}
			}//for 1 end
			
			
			//第二步 后向传播误差
			//2 backward error 
			GLayerFull* outLayer = (GLayerFull*)m_layerPtrVector[layerCount-1] ;
			//计算输出层误差数据 使用cross entry 和 softmax
			//ok
			#ifndef USE_GPU_MODE
			cpu_full_backwardErrorFromLabel(
				pData->m_label ,
				outLayer->m_actiArray->getHostMemory() ,
				outLayer->m_errorArray->getHostMemory() ,
				outLayer->m_actiArray->getNFloat()
			)  ;
			//outLayer->m_errorArray->copyHost2Device() ;
				#ifdef USE_DEBUG_OUTPUT
				writeDebugFloatArray("output error" , layerCount-1 , 
					outLayer->m_errorArray->getHostMemory() , 
					outLayer->m_errorArray->getNFloat() , 
					1 , 
					1 , 
					1 ) ;
				#endif
			#else 
			gpu_full_backwardErrorFromLabel
			<<<1,512>>>(
				pData->m_label ,
				outLayer->m_actiArray->getDevMemory() ,
				outLayer->m_errorArray->getDevMemory() ,
				outLayer->m_actiArray->getNFloat()
			) ;
			//cudaDeviceSynchronize();
			#endif
			//ok

			//compute output error. 其他层后向传递误差
			
			for(int ilayer = layerCount-2 ; ilayer >= 0 ; -- ilayer ){
				
				GLayer* currLayer = m_layerPtrVector[ilayer]  ;
				GLayer* nextLayer = m_layerPtrVector[ilayer+1] ;
				
				if( currLayer->getType()==GLayerTypeFull && 
					nextLayer->getType()==GLayerTypeFull ){
						
					GLayerFull* currlayer1 = (GLayerFull*)currLayer ;
					GLayerFull* nextlayer1 = (GLayerFull*)nextLayer ;
					//后向传递误差error 20170607
					//ok
					#ifndef USE_GPU_MODE
					cpu_full_backwardErrorFromNextLayer(
												nextlayer1->m_errorArray->getHostMemory() , 
												nextlayer1->m_biasAndWeights->getHostMemory()  , 
												nextlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_actiArray->getHostMemory() ,
												currlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_errorArray->getHostMemory() ,
												currlayer1->m_dropoutMaskArray->getHostMemory()
												) ;
						#ifdef USE_DEBUG_OUTPUT
						writeDebugFloatArray("full error" , ilayer , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							1 , 
							1 , 
							1 ) ;
						#endif
					#else
					gpu_full_backwardErrorFromNextLayer
					<<<(currlayer1->m_errorArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
					(
												nextlayer1->m_errorArray->getDevMemory() , 
												nextlayer1->m_biasAndWeights->getDevMemory()  , 
												nextlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_actiArray->getDevMemory() ,
												currlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_errorArray->getDevMemory() ,
												currlayer1->m_dropoutMaskArray->getDevMemory()
												) ;
					#endif
					//ok
					
				}else if( currLayer->getType()==GLayerTypePool && 
							nextLayer->getType()==GLayerTypeFull ){
					//全连接后向传递到池层
					GLayerPool* currlayer1 = (GLayerPool*)currLayer ;
					GLayerFull* nextlayer1 = (GLayerFull*)nextLayer ;
					#ifndef USE_GPU_MODE
					cpu_full_backwardErrorFromNextLayer(
												nextlayer1->m_errorArray->getHostMemory() , 
												nextlayer1->m_biasAndWeights->getHostMemory()  , 
												nextlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_actiArray->getHostMemory() ,
												currlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_errorArray->getHostMemory() ,
												0
												) ;
						#ifdef USE_DEBUG_OUTPUT
						writeDebugFloatArray("pool error" , ilayer , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->outXSize , 
							currlayer1->outYSize , 
							currlayer1->bandCount , 
							1 ) ;
						#endif
					#else
					gpu_full_backwardErrorFromNextLayer
					<<<(currlayer1->m_errorArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
					(
												nextlayer1->m_errorArray->getDevMemory() , 
												nextlayer1->m_biasAndWeights->getDevMemory()  , 
												nextlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_actiArray->getDevMemory() ,
												currlayer1->m_actiArray->getNFloat() , 
												currlayer1->m_errorArray->getDevMemory() ,
												0
												) ;
					#endif
								
								
				}else if( currLayer->getType()==GLayerTypeConv &&
				nextLayer->getType()==GLayerTypePool ){
					//池层后向传递到卷积层
					GLayerConv* currlayer1 = (GLayerConv*)currLayer ;
					GLayerPool* nextlayer1 = (GLayerPool*)nextLayer ;
					
					#ifndef USE_GPU_MODE
					cpu_conv_backwardErrorFromPoolLayer
					(
						nextlayer1->m_errorArray->getHostMemory() , 
						nextlayer1->m_convIsMaxArray->getHostMemory() , 
						nextlayer1->outXSize * nextlayer1->outYSize , 
						nextlayer1->outXSize  , 
						currlayer1->m_actiArray->getNFloat() , 
						currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
						currlayer1->m_ioXsize , 
						currlayer1->m_errorArray->getHostMemory() 
					) ;
						#ifdef USE_DEBUG_OUTPUT
						writeDebugFloatArray("conv error" , ilayer , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_ioXsize , 
							currlayer1->m_ioXsize , 
							currlayer1->m_kernelCount , 
							1 ) ;
						#endif
					#else
					gpu_conv_backwardErrorFromPoolLayer
					<<<(currlayer1->m_errorArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
					(
						nextlayer1->m_errorArray->getDevMemory() , 
						nextlayer1->m_convIsMaxArray->getDevMemory() , 
						nextlayer1->outXSize * nextlayer1->outYSize , 
						nextlayer1->outXSize  , 
						currlayer1->m_actiArray->getNFloat() , 
						currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
						currlayer1->m_ioXsize , 
						currlayer1->m_errorArray->getDevMemory() 
					) ;
					//cudaDeviceSynchronize();
					#endif
				}else if( currLayer->getType()==GLayerTypePool &&
				nextLayer->getType()==GLayerTypeConv ) {
					GLayerPool* currlayer1 = (GLayerPool*)currLayer ;
					GLayerConv* nextlayer1 = (GLayerConv*)nextLayer ;
					#ifndef USE_GPU_MODE
					cpu_image_backwardErrorFromConvLayer(
						nextlayer1->m_errorArray->getHostMemory()  , 
						nextlayer1->m_actiArray->getHostMemory() , 
						nextlayer1->m_kernelWeightsBiasArray->getHostMemory(), 
						nextlayer1->m_ioXsize  , 
						nextlayer1->m_ioYsize  ,
						nextlayer1->m_ioXsize * nextlayer1->m_ioYsize , 
						nextlayer1->m_kXsize * nextlayer1->m_kXsize * nextlayer1->m_inBandCount ,
						nextlayer1->m_kXsize/2 ,
						nextlayer1->m_kXsize , 
						nextlayer1->m_kernelCount , 
						currlayer1->m_errorArray->getNFloat() , 
						currlayer1->outXSize * currlayer1->outYSize , 
						currlayer1->outXSize  , 
						currlayer1->m_errorArray->getHostMemory() 
					) ;
						#ifdef USE_DEBUG_OUTPUT
						writeDebugFloatArray("pool error" , ilayer , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->outXSize , 
							currlayer1->outYSize , 
							currlayer1->bandCount , 
							1 ) ;
						#endif
					#else
					gpu_image_backwardErrorFromConvLayer
					<<<(currlayer1->m_errorArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,this->m_nthreadPerBlock>>>
					(
						nextlayer1->m_errorArray->getDevMemory()  , 
						nextlayer1->m_actiArray->getDevMemory() , 
						nextlayer1->m_kernelWeightsBiasArray->getDevMemory(), 
						nextlayer1->m_ioXsize  , 
						nextlayer1->m_ioYsize  ,
						nextlayer1->m_ioXsize * nextlayer1->m_ioYsize , 
						nextlayer1->m_kXsize * nextlayer1->m_kXsize * nextlayer1->m_inBandCount ,
						nextlayer1->m_kXsize/2 ,
						nextlayer1->m_kXsize , 
						nextlayer1->m_kernelCount , 
						currlayer1->m_errorArray->getNFloat() , 
						currlayer1->outXSize * currlayer1->outYSize , 
						currlayer1->outXSize  , 
						currlayer1->m_errorArray->getDevMemory() 
					) ;						
					//cudaDeviceSynchronize();
					#endif
					
				}
				else{
					std::cout<<"bad backward from "<<nextLayer->getType()
					<<" to "<<currLayer->getType()<<std::endl ;
					exit(3) ;
				}
				
			}

			//第三步  计算bias和weights变化，并在batch内求和
			//for 3 compute and sum changes of bias and weights
			//从前到后计算每个样本的bias和weights的变化，并加和到sum数组里面20170608
			for(int ilayer = 0 ; ilayer < layerCount ; ++ ilayer ){
				GLayer* currLayer = m_layerPtrVector[ilayer]  ;
				
				if( ilayer == 0 ){
					//第一层使用输入数据的值作为激励值计算bias和weights的变化
					if( currLayer->getType()==GLayerTypeFull ){
						GLayerFull* currlayer1 = (GLayerFull*)currLayer ;
						#ifndef USE_GPU_MODE
						cpu_full_computeAndSumBiasAndWeightsChanges(
							pData->m_dataPtr->getHostMemory() , 
							pData->m_dataPtr->getNFloat() , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getHostMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("full-dwsum,prev-input" , ilayer , 
								currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize+1 , 
								1 , 
								1 ) ;//0709
							#endif
						#else
						gpu_full_computeAndSumBiasAndWeightsChanges
						<<<(currlayer1->m_biasAndWeightsChangesSum->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							pData->m_dataPtr->getDevMemory() , 
							pData->m_dataPtr->getNFloat() , 
							currlayer1->m_errorArray->getDevMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getDevMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
						//cudaDeviceSynchronize();
						#endif
					}else if( currLayer->getType()==GLayerTypeConv ){
						GLayerConv* currlayer1 = (GLayerConv*)currLayer ;
						#ifndef USE_GPU_MODE
						cpu_conv_computeAndSumKernelWeightsChanges
						(
							pData->m_dataPtr->getHostMemory() , 
							currlayer1->m_ioXsize, 
							currlayer1->m_ioYsize , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
							currlayer1->m_kernelCount , 
							currlayer1->m_kXsize , 
							currlayer1->m_kXsize/2 , 
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount ,
							currlayer1->m_kXsize * currlayer1->m_kXsize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() ,
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount * currlayer1->m_kernelCount
						) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("conv-dwsum,prev-input" , ilayer , 
								currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							#endif
						
						#else
						gpu_conv_computeAndSumKernelWeightsChanges
						<<<(currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() +this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							pData->m_dataPtr->getDevMemory() , 
							currlayer1->m_ioXsize, 
							currlayer1->m_ioYsize , 
							currlayer1->m_errorArray->getDevMemory() , 
							currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
							currlayer1->m_kernelCount , 
							currlayer1->m_kXsize , 
							currlayer1->m_kXsize/2 , 
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount ,
							currlayer1->m_kXsize * currlayer1->m_kXsize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getDevMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() ,
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount * currlayer1->m_kernelCount
						)  ;
						//cudaDeviceSynchronize();
						#endif
					}else{
						std::cout<<"bad sum 0"<<std::endl ;
						exit(5) ;
					}
					
				}else{
					//其他层使用前一层的激励值计算bias和weights的变化
					
					//   20170626-1056
					GLayer* prevLayer = m_layerPtrVector[ilayer-1] ;
					if( currLayer->getType()==GLayerTypeFull 
						&& prevLayer->getType()==GLayerTypeFull
					){
						GLayerFull* currlayer1 = (GLayerFull*)currLayer ;
						GLayerFull* prevlayer1 = (GLayerFull*)prevLayer ;
						#ifndef USE_GPU_MODE
						cpu_full_computeAndSumBiasAndWeightsChanges(
							prevlayer1->m_actiArray->getHostMemory() , 
							prevlayer1->m_actiArray->getNFloat() , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getHostMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("full-dwsum,prev-full" , ilayer , 
								currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize+1 , 
								1 , 
								1 ) ;//
							#endif
						#else
						gpu_full_computeAndSumBiasAndWeightsChanges
						<<<(currlayer1->m_biasAndWeightsChangesSum->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							prevlayer1->m_actiArray->getDevMemory() , 
							prevlayer1->m_actiArray->getNFloat() , 
							currlayer1->m_errorArray->getDevMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getDevMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
						//cudaDeviceSynchronize();
						#endif
						
					}
					else if( currLayer->getType()==GLayerTypeFull 
						&& prevLayer->getType()==GLayerTypePool
					){
						GLayerFull* currlayer1 = (GLayerFull*)currLayer ;
						GLayerPool* prevlayer1 = (GLayerPool*)prevLayer ;
						#ifndef USE_GPU_MODE
						cpu_full_computeAndSumBiasAndWeightsChanges(
							prevlayer1->m_actiArray->getHostMemory() , 
							prevlayer1->m_actiArray->getNFloat() , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getHostMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
								#ifdef USE_DEBUG_OUTPUT
								writeDebugFloatArray("full-dwsum,prev-pool" , ilayer , 
									currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
									currlayer1->m_outsize , 
									currlayer1->m_insize+1 , 
									1 , 
									1 ) ;//
								#endif
						#else
						gpu_full_computeAndSumBiasAndWeightsChanges
						<<<(currlayer1->m_biasAndWeightsChangesSum->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							prevlayer1->m_actiArray->getDevMemory() , 
							prevlayer1->m_actiArray->getNFloat() , 
							currlayer1->m_errorArray->getDevMemory() , 
							currlayer1->m_errorArray->getNFloat() , 
							currlayer1->m_biasAndWeightsChangesSum->getDevMemory() ,
							currlayer1->m_biasAndWeightsChangesSum->getNFloat()
							) ;
						//cudaDeviceSynchronize();
						#endif
					}
					else if( currLayer->getType()==GLayerTypeConv 
						&& prevLayer->getType()==GLayerTypePool
					){
						GLayerConv* currlayer1 = (GLayerConv*)currLayer ;
						GLayerPool* prevlayer1 = (GLayerPool*)prevLayer ;
						
						#ifndef USE_GPU_MODE
						cpu_conv_computeAndSumKernelWeightsChanges
						(
							prevlayer1->m_actiArray->getHostMemory() , 
							currlayer1->m_ioXsize, 
							currlayer1->m_ioYsize , 
							currlayer1->m_errorArray->getHostMemory() , 
							currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
							currlayer1->m_kernelCount , 
							currlayer1->m_kXsize , 
							currlayer1->m_kXsize/2 , 
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount ,
							currlayer1->m_kXsize * currlayer1->m_kXsize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() ,
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount * currlayer1->m_kernelCount
						) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("conv-dwsum,prev-pool" , ilayer , 
								currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							#endif

						#else
						gpu_conv_computeAndSumKernelWeightsChanges
						<<<(currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() +this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							prevlayer1->m_actiArray->getDevMemory() , 
							currlayer1->m_ioXsize, 
							currlayer1->m_ioYsize , 
							currlayer1->m_errorArray->getDevMemory() , 
							currlayer1->m_ioXsize * currlayer1->m_ioYsize , 
							currlayer1->m_kernelCount , 
							currlayer1->m_kXsize , 
							currlayer1->m_kXsize/2 , 
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount ,
							currlayer1->m_kXsize * currlayer1->m_kXsize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getDevMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() ,
							currlayer1->m_kXsize * currlayer1->m_kXsize * currlayer1->m_inBandCount * currlayer1->m_kernelCount
						)  ;
						//cudaDeviceSynchronize();
						#endif
						
					}

				}
			}// end for 3 

			++ ibat ;
			++ idata ;
			//第四步 是否达到批次的size，达到批次size计算bias和weights变化平均值，
			// 根据studyrate和momentum更新bias和weights
			//4 每当计算的数据达到批次Size的时候，
			//计算bias和weights的变化平均值并更新bias和weights
			if( ibat == batchSize ){
				#ifdef USE_DEBUG_OUTPUT
				writeDebugLine("ibat==batchSize , ibat:" ,ibat ) ;
				writeDebugLine("batchsize:" , batchSize) ;
				writeDebugLineFloat("study " , studyRate) ;
				writeDebugLineFloat("momentum " , momentum) ;
				#endif
				ibat = 0 ;
				//5-1 update bias and weights.
				//对每个层次更新bias和weights
				for(int ilayer = 0 ; ilayer < layerCount ; ++ ilayer ){
					GLayer* currLayer = m_layerPtrVector[ilayer]  ;
					
					if( currLayer->getType()==GLayerTypeFull ){
						GLayerFull* currlayer1 = (GLayerFull*)currLayer ;
						#ifndef USE_GPU_MODE
							#ifdef USE_DEBUG_OUTPUT 
							writeDebugFloatArray("before updated full last changes dw" , ilayer , 
								currlayer1->m_lastBiasAndWeightsChanges->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1  , 
								1 , 
								1 ) ;//
							writeDebugFloatArray("before update full dwsum" , ilayer , 
								currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1, 
								1 , 
								1 ) ;//
							writeDebugFloatArray("before updated full updated dw" , ilayer , 
								currlayer1->m_biasAndWeights->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1  , 
								1 , 
								1 ) ;//
							#endif
						cpu_updateBiasAndWeights (
							studyRate , 
							momentum , 
							batchSize , 
							currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
							currlayer1->m_biasAndWeightsChangesSum->getNFloat() , 
							currlayer1->m_lastBiasAndWeightsChanges->getHostMemory() , 
							currlayer1->m_biasAndWeights->getHostMemory() 
							) ;
							#ifdef USE_DEBUG_OUTPUT 
							writeDebugFloatArray("after updated full last changes dw" , ilayer , 
								currlayer1->m_lastBiasAndWeightsChanges->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1  , 
								1 , 
								1 ) ;//
							writeDebugFloatArray("after update full dwsum" , ilayer , 
								currlayer1->m_biasAndWeightsChangesSum->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1, 
								1 , 
								1 ) ;//
							writeDebugFloatArray("after updated full updated dw" , ilayer , 
								currlayer1->m_biasAndWeights->getHostMemory() , 
								currlayer1->m_outsize , 
								currlayer1->m_insize + 1  , 
								1 , 
								1 ) ;//
							
							#endif
							
						#else
						gpu_updateBiasAndWeights
						<<<(currlayer1->m_biasAndWeightsChangesSum->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							studyRate , 
							momentum , 
							batchSize , 
							currlayer1->m_biasAndWeightsChangesSum->getDevMemory() , 
							currlayer1->m_biasAndWeightsChangesSum->getNFloat() , 
							currlayer1->m_lastBiasAndWeightsChanges->getDevMemory() , 
							currlayer1->m_biasAndWeights->getDevMemory() 
							) ;
						#endif
						
						//dropout
						if( currlayer1->m_useDropoutMask ){
							currlayer1->shuffleDropoutMaskArray() ;
						}
						
					}else if(currLayer->getType()==GLayerTypeConv) {
						GLayerConv* currlayer1 = (GLayerConv*)currLayer ;
						#ifndef USE_GPU_MODE
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("before updated conv last changes dw" , ilayer , 
								currlayer1->m_kernelWeightsBiasLastChangeArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							writeDebugFloatArray("before update conv dwsum" , ilayer , 
								currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount  ) ;//
							writeDebugFloatArray("before updated conv updated dw" , ilayer , 
								currlayer1->m_kernelWeightsBiasArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							
							#endif
							
							#ifdef USE_DEBUG_OUTPUT2 
							fprintf(g_debug_output_fileptr2 , 
							"before update weights w0:%8.4f w1:%8.4f w2:%8.4f\n" , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[0] , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[1] , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[2] 
							) ;
							#endif
						cpu_updateBiasAndWeights (
							studyRate , 
							momentum , 
							batchSize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() , 
							currlayer1->m_kernelWeightsBiasLastChangeArray->getHostMemory() , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory() 
							) ;
							#ifdef USE_DEBUG_OUTPUT
							writeDebugFloatArray("after updated conv last changes dw" , ilayer , 
								currlayer1->m_kernelWeightsBiasLastChangeArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							writeDebugFloatArray("after update conv dwsum" , ilayer , 
								currlayer1->m_kernelWeightsBiasChangeSumArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount  ) ;//
							writeDebugFloatArray("after updated conv updated dw" , ilayer , 
								currlayer1->m_kernelWeightsBiasArray->getHostMemory() , 
								currlayer1->m_kXsize , 
								currlayer1->m_kXsize , 
								currlayer1->m_inBandCount , 
								currlayer1->m_kernelCount ) ;//
							
							#endif
							
							#ifdef USE_DEBUG_OUTPUT2 
							fprintf(g_debug_output_fileptr2 , 
							"After  update weights w0:%8.4f w1:%8.4f w2:%8.4f\n" , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[0] , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[1] , 
							currlayer1->m_kernelWeightsBiasArray->getHostMemory()[2] 
							) ;
							#endif
						#else
						gpu_updateBiasAndWeights
						<<<(currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat()+this->m_nthreadPerBlock-1)/this->m_nthreadPerBlock,
						this->m_nthreadPerBlock>>>
						(
							studyRate , 
							momentum , 
							batchSize , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getDevMemory() , 
							currlayer1->m_kernelWeightsBiasChangeSumArray->getNFloat() , 
							currlayer1->m_kernelWeightsBiasLastChangeArray->getDevMemory() , 
							currlayer1->m_kernelWeightsBiasArray->getDevMemory()
						) ;
						//cudaDeviceSynchronize();
						#endif
					}
					
					
				}

				#ifdef USE_DEBUG_OUTPUT
				static int static_debug_output_n = 0 ;
				static_debug_output_n ++ ;
				if( static_debug_output_n==5 ){
					exit(0) ;
				}
				#endif

				//jump out this repo. 检查下个批次是否超出总样本数量
				//如果超过总样本数，结束本次repo，进入下一个repo
				if( idata + batchSize > dataCount ){
					break ;
				}
			}// end 5

		}//end white for one repo
		
		//输出该次repos的总mse msetotal
		
		float weight0 = 0.0f ;
		float weight1 = 0.0f ;
		#ifdef USE_GPU_MODE
		goodBadMseArray.copyDeviceToHost() ;
		repoGood = goodBadMseArray.getHostMemory()[0] ;
		repoBad = goodBadMseArray.getHostMemory()[1] ;
		repoMse = goodBadMseArray.getHostMemory()[2] ;
		weight0 = goodBadMseArray.getHostMemory()[3] ;
		weight1 = goodBadMseArray.getHostMemory()[4] ;
		#else
		GLayer* templayer0 = m_layerPtrVector[0] ;
		if( templayer0->getType() == GLayerTypeConv ){
			GLayerConv* templayer00 = (GLayerConv*)templayer0 ;
			weight0 = templayer00->m_kernelWeightsBiasArray->getHostMemory()[0] ;
			weight1 = templayer00->m_kernelWeightsBiasArray->getHostMemory()[1] ;
		}
		#endif
				
		float repoGoodPercent = repoGood*1.0f / (repoGood+repoBad) ;
		float repoBadPercent = repoBad*1.0f / (repoGood+repoBad) ;
		std::cout<<"ir:"<<irepo<<" ; mse:"<<repoMse
		<<" ; nG%:"<<std::setprecision(4)<<repoGoodPercent
		<<" ; nB%:"<<std::setprecision(4)<<repoBadPercent
		<<" ; w0:"<<std::setprecision(4)<<weight0
		<<" ; w1:"<<std::setprecision(4)<<weight1
		<<std::endl ;
		
		#ifdef USE_DEBUG_OUTPUT2
		fprintf(g_debug_output_fileptr2 , "end irepo:%d mse:%8.4f nG:%8.4f nB:%8.4f\n" , 
		irepo , repoMse , repoGoodPercent , repoBadPercent ) ;
		#endif
		
		//repo mse小于阈值退出训练
		if( repoMse < finishMse ){
			std::cout<<"repoMse lower than finishMse." 
						" Training Finished."<<std::endl;
			std::string filepathForSaving = currentDateTimeString( std::string("-") ) ;
			this->saveToFile(filepathForSaving.c_str() , 100) ;  
			break ;
		}else{
			// repo end timestamp
			unsigned endrepoTimestamp = (unsigned)time(NULL) ;
			if( endrepoTimestamp - saveTimestamp > (unsigned)this->m_saveSeconds ){
				saveTimestamp = endrepoTimestamp  ;
				string filepathForSaving = currentDateTimeString(std::string("-") ) ;
				this->saveToFile(filepathForSaving.c_str() , 0) ;
			}
		}
		
	}//end for repos
	
	//将显存中的b和w拷贝回内存
	for(int ilayer = 0 ; ilayer < (int)m_layerPtrVector.size() ; ++ ilayer ){
		GLayer* layer = m_layerPtrVector[ilayer] ;
		if( layer->getType()==GLayerTypeFull ){
			GLayerFull* tlayer = (GLayerFull*)layer ;
			tlayer->m_biasAndWeights->copyDeviceToHost() ;
		}else if( layer->getType()==GLayerTypeConv ){
			GLayerConv* tlayer = (GLayerConv*)layer ;
			tlayer->m_kernelWeightsBiasArray->copyDeviceToHost() ;
		}
	}
	
}





void GConvNetwork::saveToFile( const char* filepath , int fileid ) {
	
	char buff[1024] ;
	sprintf(buff , "%s%d.json" , filepath , fileid ) ;
	std::cout<<"saving into "<<std::string(buff)<<std::endl ;
	
	Json::Value root ;
	root["study-rate"] = m_studyRate ;
	root["momentum"] = m_momentum ;
	root["batch-size"] = m_batchSize ;
	root["num-repos"] = m_numRepos ;
	root["finish-mse"] = m_finishMse ;
	root["save-sec"] = m_saveSeconds ;
	root["num-thread-per-block"] = m_nthreadPerBlock ;
	root["data-scale"] = m_dataScale ;
	
	for(int i = 0 ; i<(int)this->m_layerPtrVector.size() ; ++ i ){
		Json::Value layerNode = this->m_layerPtrVector[i]->toJsonNode() ;
		root["layers"][i] = layerNode ;
	}
	
	std::ofstream outfileid;
    outfileid.open(buff);
	Json::StyledWriter styledWriter;
    outfileid << styledWriter.write(root);

    outfileid.close();

}



void GConvNetwork::loadFromFile( const char* filepath ) {
	
	std::string theFilePath ;
	if( filepath == 0 ){
		theFilePath = std::string("netsetup.json") ;
	}else{
		theFilePath = std::string(filepath) ;
	}
	
	std::cout<<"loading network from "<<theFilePath<<std::endl ;
	
	Json::Value root;

	std::ifstream file(theFilePath);
	file >> root;
	file.close() ;
	
	m_studyRate     = root["study-rate"].asFloat() ;// gfile.readLabelFloatValue("#study-rate" , 0 ) ;
	m_momentum      = root["momentum"].asFloat() ;//gfile.readLabelFloatValue("#momentum" , 0 ) ;
	m_batchSize     = root["batch-size"].asInt() ;//gfile.readLabelIntValue("#batch-size" , 0 ) ;
	m_numRepos      = root["num-repos"].asInt() ;//gfile.readLabelIntValue("#num-repos" , 0 ) ;
	m_finishMse     = root["finish-mse"].asFloat() ;//gfile.readLabelFloatValue("#finish-mse" , 0 ) ;
    m_saveSeconds   = root["save-sec"].asInt() ;//gfile.readLabelIntValue("#save-sec" , 0 ) ;
	m_nthreadPerBlock = root["num-thread-per-block"].asInt() ;//gfile.readLabelIntValue("#num-thread-per-block" , 0 ) ;
	m_dataScale       = root["data-scale"].asFloat() ;//gfile.readLabelFloatValue("#data-scale" , 0 ) ;
	
	std::cout<<"m_studyRate:"<<m_studyRate<<std::endl ;
	std::cout<<"m_momentum:"<<m_momentum<<std::endl ;
	std::cout<<"m_batchSize:"<<m_batchSize<<std::endl ;
	std::cout<<"m_numRepos:"<<m_numRepos<<std::endl ;
	std::cout<<"m_finishMse:"<<m_finishMse<<std::endl ;
	std::cout<<"m_saveSeconds:"<<m_saveSeconds<<std::endl ;
	std::cout<<"m_nthreadPerBlock:"<<m_nthreadPerBlock<<std::endl ;
	std::cout<<"m_dataScale:"<<m_dataScale<<std::endl ;
	
	int numberOfLayers = 0 ;
	numberOfLayers = root["layers"].size() ;
	std::cout<<"Number of Layers:"<<numberOfLayers<<std::endl ;
	for(int i = 0 ; i < numberOfLayers ; ++ i ){
		Json::Value layerNode = root["layers"][i] ;
		GLayerType layerType = (GLayerType) layerNode["layer-type"].asInt() ;
		if( layerType == GLayerTypeConv ){
			GLayerConv* layerPtr = new GLayerConv(layerNode) ;
			this->m_layerPtrVector.push_back(layerPtr) ;
		}else if( layerType == GLayerTypeFull ){
			GLayerFull* layerPtr = new GLayerFull(layerNode) ;
			this->m_layerPtrVector.push_back(layerPtr) ;
		}else if( layerType == GLayerTypePool ){
			GLayerPool* layerPtr = new GLayerPool(layerNode) ;
			this->m_layerPtrVector.push_back(layerPtr) ;
		}
	}
}

int GConvNetwork::guess( float* dataArray , int dsize , float* possibility ) {
	//1 forward 不设输入层 
	int layerCount = (int) this->m_layerPtrVector.size() ;
	for(int ilayer = 0 ; ilayer < layerCount ; ++ ilayer ){
		GLayer* layerPtr = m_layerPtrVector[ilayer] ;
		if( ilayer == 0 ){
			 //第一层直接获得输入数据
			 if( layerPtr->getType()==GLayerTypeConv ){ 
				 //卷积层
				 GLayerConv* tlayer = (GLayerConv*)layerPtr ;
				 cpu_conv_forwardFromImage(	
					 dataArray  , 
					 tlayer->m_ioXsize, 
					 tlayer->m_ioYsize , 
					 tlayer->m_inBandCount ,
					 tlayer->m_kernelWeightsBiasArray->getHostMemory()  , 
					 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
					 tlayer->m_actiArray->getHostMemory() , 
					 tlayer->m_reluArray->getHostMemory() , 
					 tlayer->m_kXsize , 
					 tlayer->m_kXsize/2 , 
					 tlayer->m_kernelCount  , 
					 tlayer->m_ioXsize*tlayer->m_ioYsize , 
					 tlayer->m_actiArray->getNFloat() ) ;

			 }else if( layerPtr->getType()==GLayerTypeFull ){
				 //全连接层  
				 GLayerFull* tlayer = (GLayerFull*)layerPtr ;
				 tlayer->setAllMaskOne() ;
				 cpu_full_forwardFromPrevLayer(
									dataArray  ,
									dsize ,
									tlayer->m_biasAndWeights->getHostMemory() ,
									tlayer->m_actiArray->getHostMemory() ,
									tlayer->m_actiArray->getNFloat() ,
									tlayer->m_dropoutMaskArray->getHostMemory() 
									) ;
			 }else{
				 std::cout<<"bad layer 0  guess"<<std::endl ;
				 exit(1) ;
			 }
			
		}else{
			
			//后面网络层使用前一层的激励值作为当前输入
			GLayer* prevLayer = m_layerPtrVector[ilayer-1] ;
			
			if( layerPtr->getType() == GLayerTypeConv 
				&& prevLayer->getType()==GLayerTypePool ){
				//卷积层计算
				GLayerPool* player = (GLayerPool*)prevLayer ;
				GLayerConv* tlayer = (GLayerConv*)layerPtr ;
				cpu_conv_forwardFromImage(
					player->m_actiArray->getHostMemory() , 
					 tlayer->m_ioXsize, 
					 tlayer->m_ioYsize , 
					 tlayer->m_inBandCount ,
					 tlayer->m_kernelWeightsBiasArray->getHostMemory()  , 
					 tlayer->m_kXsize*tlayer->m_kYsize*tlayer->m_inBandCount*tlayer->m_kernelCount , 
					 tlayer->m_actiArray->getHostMemory() , 
					 tlayer->m_reluArray->getHostMemory() , 
					 tlayer->m_kXsize , 
					 tlayer->m_kXsize/2 , 
					 tlayer->m_kernelCount  , 
					 tlayer->m_ioXsize*tlayer->m_ioYsize , 
					 tlayer->m_actiArray->getNFloat() 
				) ;				
				
			}else if( layerPtr->getType() == GLayerTypeFull 
						&& prevLayer->getType()==GLayerTypePool ) {
				//全连接层计算
				GLayerPool* player = (GLayerPool*)prevLayer ;
				GLayerFull* tlayer = (GLayerFull*)layerPtr ;
				if( ilayer == layerCount-1 ){
					cpu_output_forwardFromPrevLayer(
									player->m_actiArray->getHostMemory() ,
									player->m_actiArray->getNFloat() ,
									tlayer->m_biasAndWeights->getHostMemory() ,
									tlayer->m_actiArray->getHostMemory() ,
									tlayer->m_actiArray->getNFloat()
									) ;
				}else{

					tlayer->setAllMaskOne() ;
					cpu_full_forwardFromPrevLayer(
									player->m_actiArray->getHostMemory() ,
									player->m_actiArray->getNFloat() ,
									tlayer->m_biasAndWeights->getHostMemory() ,
									tlayer->m_actiArray->getHostMemory() ,
									tlayer->m_actiArray->getNFloat(), 
									tlayer->m_dropoutMaskArray->getHostMemory()
									) ;
				}
				
				
			}else if( layerPtr->getType() == GLayerTypeFull 
						&& prevLayer->getType()==GLayerTypeFull ) {
				//全连接层计算
				GLayerFull* player = (GLayerFull*)prevLayer ;
				GLayerFull* tlayer = (GLayerFull*)layerPtr ;
				if( ilayer == layerCount -1 ){
					cpu_output_forwardFromPrevLayer(
									player->m_actiArray->getHostMemory() ,
									player->m_actiArray->getNFloat() ,
									tlayer->m_biasAndWeights->getHostMemory() ,
									tlayer->m_actiArray->getHostMemory() ,
									tlayer->m_actiArray->getNFloat()
									) ;
				}else{
					tlayer->setAllMaskOne() ;
					cpu_full_forwardFromPrevLayer(
									player->m_actiArray->getHostMemory() ,
									player->m_actiArray->getNFloat() ,
									tlayer->m_biasAndWeights->getHostMemory() ,
									tlayer->m_actiArray->getHostMemory() ,
									tlayer->m_actiArray->getNFloat() , 
									tlayer->m_dropoutMaskArray->getHostMemory()
									) ;
				}				
			}else if( layerPtr->getType() == GLayerTypePool 
						&& prevLayer->getType()==GLayerTypeConv ) {
				//池层计算
				GLayerConv* player = (GLayerConv*)prevLayer ;
				GLayerPool* tlayer = (GLayerPool*)layerPtr ;

				cpu_pool_forwardFromImage(
					player->m_actiArray->getHostMemory() , 
					player->m_ioXsize , 
					player->m_ioYsize , 
					player->m_kernelCount , 
					player->m_ioXsize * player->m_ioYsize , 
					tlayer->m_actiArray->getHostMemory() , 
					tlayer->m_convIsMaxArray->getHostMemory() , 
					tlayer->outXSize , 
					tlayer->outYSize , 
					tlayer->outXSize * tlayer->outYSize , 
					tlayer->m_actiArray->getNFloat() ) ;
								
			}else{
				std::cout<<"bad layer guess forward from"<<
					layerPtr->getType()<<" to "<<
					prevLayer->getType()
					<<std::endl ;
				exit(2) ;
			}
			
			if( ilayer == layerCount-1 ){
				GLayerFull* outlayer = (GLayerFull*)layerPtr ;
				float outposs = 0.0f ;
				int iguess = 
				cpu_output_computeOutputSoftmaxValues(
					outlayer->m_actiArray->getHostMemory() , 
					outlayer->m_actiArray->getNFloat() , 
					&outposs ) ;
				*possibility = outposs ;
				return iguess ;
			} 
		}
	}
	return -1 ;
}


//对特定像素 pixelIndex 数组进行反卷积可视化
void GConvNetwork::visualizeFromActivationValueArray( const char* prefix, int sampleid , int theLayerIndex , int* pixelIndexArr , float* pixelValueArr , int arrSize )  {
	
	GLayer* theLayerPtr = this->m_layerPtrVector[theLayerIndex] ;
	if( theLayerPtr->getType() != GLayerTypeConv && theLayerPtr->getType() != GLayerTypePool  ){
		std::cout<<"not a conv , pool layer , could not be visualized."<<std::endl ;
		return ;
	}
	
	//set all activations zero but pixelIndex
	/*
	if( theLayerPtr->getType() == GLayerTypeConv ){
		GLayerConv* theConvLayer = (GLayerConv*)theLayerPtr ;
		for(int it =  0 ; it<theConvLayer->m_actiArray->getNFloat() ; ++ it  ){
			for(int ia = 0 ; ia < arrSize ; ++ ia ){
				if( it != pixelIndexArr[ia] ){
					theConvLayer->m_actiArray->getHostMemory()[it] = 0.0f ;
				}else{
					//relu
					theConvLayer->m_actiArray->getHostMemory()[it] = fmaxf(0.0f , pixelValueArr[ia] );
					break ;
				}
			}
		}
	}else{
		GLayerPool* thePoolLayer = (GLayerPool*)theLayerPtr ;
		for(int it =  0 ; it<thePoolLayer->m_actiArray->getNFloat() ; ++ it  ){
			for(int ia = 0 ; ia < arrSize ; ++ ia ){ 
				if( it != pixelIndexArr[ia] ){
					thePoolLayer->m_actiArray->getHostMemory()[it] = 0.0f ;
				}else{
					//relu
					thePoolLayer->m_actiArray->getHostMemory()[it] = fmaxf(0.0f , pixelValueArr[ia] );
					break ;
				}
			}
		}
	}
	 * */
	
	
	//第一个层的输入尺寸
	GLayerConv* layer0 = (GLayerConv*)this->m_layerPtrVector[0] ;
	int nfloatOfImage = layer0->m_ioXsize * layer0->m_ioYsize * layer0->m_inBandCount ;
	//可视化输出图像
	float* visualImage = new float[nfloatOfImage] ;
	for(int it = 0 ; it < nfloatOfImage ; ++ it  ){
		visualImage[it] = 0.0f ;
	}

	for(int ilayer = theLayerIndex ; ilayer >= 0 ; -- ilayer ){
		GLayer* layerPtr = this->m_layerPtrVector[ilayer] ;
		if( layerPtr->getType() == GLayerTypeConv ){
			GLayerConv* clayer = (GLayerConv*)layerPtr ;
			if( ilayer == 0 ){
				unconvFromConvToImage( clayer , visualImage ) ;
			}else{
				GLayerPool* player = (GLayerPool*)this->m_layerPtrVector[ilayer-1] ;
				unconvFromConvToImage( clayer , player->m_actiArray->getHostMemory() ) ;
			}
		}else if( layerPtr->getType() == GLayerTypePool ) {
			GLayerPool* player = (GLayerPool*)layerPtr ;
			GLayerConv* clayer = (GLayerConv*)this->m_layerPtrVector[ilayer-1] ;
			unpoolingFromPoolToConv( player , clayer->m_actiArray->getHostMemory() ) ;
		}
	}
	
	writeArrayToFile( prefix , "-" , sampleid , "-p" , 999 , ".ppm" , visualImage , layer0->m_ioXsize , 
	layer0->m_ioYsize , layer0->m_inBandCount) ;
	
	delete [] visualImage ;
	visualImage = 0 ;
}
//对特定像素 pixelIndex 进行反卷积可视化
void GConvNetwork::visualizeFromActivationValue(int sampleid, int theLayerIndex , int pixelIndex , float pixelValue ) {
	
	GLayer* theLayerPtr = this->m_layerPtrVector[theLayerIndex] ;
	if( theLayerPtr->getType() != GLayerTypeConv && theLayerPtr->getType() != GLayerTypePool  ){
		std::cout<<"not a conv , pool layer , could not be visualized."<<std::endl ;
		return ;
	}
	
	//set all activations zero but pixelIndex
	if( theLayerPtr->getType() == GLayerTypeConv ){
		GLayerConv* theConvLayer = (GLayerConv*)theLayerPtr ;
		for(int it =  0 ; it<theConvLayer->m_actiArray->getNFloat() ; ++ it  ){
			if( it != pixelIndex ){
				theConvLayer->m_actiArray->getHostMemory()[it] = 0.0f ;
			}else{
				//relu
				theConvLayer->m_actiArray->getHostMemory()[it] = fmaxf(0.0f , pixelValue );
			}
		}
	}else{
		GLayerPool* thePoolLayer = (GLayerPool*)theLayerPtr ;
		for(int it =  0 ; it<thePoolLayer->m_actiArray->getNFloat() ; ++ it  ){
			if( it != pixelIndex ){
				thePoolLayer->m_actiArray->getHostMemory()[it] = 0.0f ;
			}else{
				//relu
				thePoolLayer->m_actiArray->getHostMemory()[it] = fmaxf(0.0f , pixelValue );
			}
		}
	}
	
	
	//第一个层的输入尺寸
	GLayerConv* layer0 = (GLayerConv*)this->m_layerPtrVector[0] ;
	int nfloatOfImage = layer0->m_ioXsize * layer0->m_ioYsize * layer0->m_inBandCount ;
	//可视化输出图像
	float* visualImage = new float[nfloatOfImage] ;
	for(int it = 0 ; it < nfloatOfImage ; ++ it  ){
		visualImage[it] = 0.0f ;
	}

	for(int ilayer = theLayerIndex ; ilayer >= 0 ; -- ilayer ){
		GLayer* layerPtr = this->m_layerPtrVector[ilayer] ;
		if( layerPtr->getType() == GLayerTypeConv ){
			GLayerConv* clayer = (GLayerConv*)layerPtr ;
			if( ilayer == 0 ){
				unconvFromConvToImage( clayer , visualImage ) ;
			}else{
				GLayerPool* player = (GLayerPool*)this->m_layerPtrVector[ilayer-1] ;
				unconvFromConvToImage( clayer , player->m_actiArray->getHostMemory() ) ;
			}
		}else if( layerPtr->getType() == GLayerTypePool ) {
			GLayerPool* player = (GLayerPool*)layerPtr ;
			GLayerConv* clayer = (GLayerConv*)this->m_layerPtrVector[ilayer-1] ;
			unpoolingFromPoolToConv( player , clayer->m_actiArray->getHostMemory() ) ;
		}
	}
	
	writeArrayToFile("vis-s" , "" , sampleid , "-p" , pixelIndex , ".ppm" , visualImage , layer0->m_ioXsize , 
	layer0->m_ioYsize , layer0->m_inBandCount) ;
	
	delete [] visualImage ;
	visualImage = 0 ;

	
}



//=======================================================================================
//=======================================================================================
//=======================================================================================




std::string currentDateTimeString( std::string ext){
	time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d.%H%M%S.log", &tstruct);
    return string(buf)+ext;
}

void writeDebugFloatArray( const char* name, int index, float* farr , int nx,int ny,int nz,int nk ){
	#ifdef USE_DEBUG_OUTPUT
	fprintf( g_debug_output_fileptr , "\n*** %s %d ***\n" , name , index ) ;
	int i = 0 ;
	for(int ik = 0 ; ik<nk ; ++ ik ){
		for(int iz = 0 ; iz < nz ; ++ iz ){
			fprintf(g_debug_output_fileptr , "* %02d %02d *\n" , ik , iz ) ;
			for(int iy = 0 ; iy < ny ; ++ iy ){
				for(int ix = 0 ; ix < nx ; ++ ix ){
					fprintf(g_debug_output_fileptr , "%8.4f " , farr[i++] ) ;
				}
				fprintf(g_debug_output_fileptr , "\n" ) ;
			}
		}
	}
	fprintf(g_debug_output_fileptr , "end arr \n" ) ;
	#endif
}
void writeDebugLine( const char* name , int index  ){
	#ifdef USE_DEBUG_OUTPUT
	fprintf( g_debug_output_fileptr , "\n--- %s %d ---\n" , name , index ) ;
	#endif
}
void writeDebugLineFloat( const char* name , float val  ){
	#ifdef USE_DEBUG_OUTPUT
	fprintf( g_debug_output_fileptr , "\n--- %s %8.4f ---\n" , name , val ) ;
	#endif
}


int main(int argc , char* argv[] ){

	std::cout<<"GPU Conv Neur Network test programe."
				" by jfwf@yeah.net 20170523."<<std::endl ;
	std::cout<<"v 0.1.20170729"<<std::endl ;
	
	#ifndef USE_GPU_MODE
	std::cout<<"Use CPU mode."<<std::endl ;
	#else
	std::cout<<"Use GPU mode!!!"<<std::endl ;
	#endif
	
	#ifdef USE_DEBUG_OUTPUT 
	std::cout<<"Use debug output !"<<std::endl ;
	g_debug_output_filename = currentDateTimeString("") ;
	std::cout<<"debug output filename:"<<g_debug_output_filename<<std::endl ;
	g_debug_output_fileptr = fopen( g_debug_output_filename.c_str() , "w" ) ;
	#endif
	
	#ifndef USE_DEBUG_OUTPUT
	srand (time(NULL));//20170710
	#endif
	
	
	#ifdef USE_DEBUG_OUTPUT2
	std::cout<<"Use debug output2 !"<<std::endl ;
	g_debug_output_filename2 = currentDateTimeString("2.log") ;
	std::cout<<"debug output2 filename:"<<g_debug_output_filename2<<std::endl ;
	g_debug_output_fileptr2 = fopen( g_debug_output_filename2.c_str() , "w" ) ;
	#endif
	
	bool enterClassifyMode = false ;
	GConvNetwork gnet ;
	std::string netsetupJsonFile = "netsetup.json" ;
	std::string samplesJsonFile = "samples.json" ;
	
	if( argc ==3 ){
		enterClassifyMode = false ;
		netsetupJsonFile = std::string(argv[1]) ;
		samplesJsonFile = std::string(argv[2]) ;
		std::cout<<"Entering training mode."<<std::endl  ;
	}else if( argc==2 ) {
		enterClassifyMode = true ;
		netsetupJsonFile = std::string(argv[1]) ;
		std::cout<<"Entering classify mode."<<std::endl  ;
	}
	std::cout<<"load netsetup file:"<<netsetupJsonFile<<std::endl ;
	gnet.loadFromFile( netsetupJsonFile.c_str() ) ;
	
	
	std::vector<std::string> labelVector  ;
	if( enterClassifyMode == false ){
		std::cout<<"loading sample images from "<<samplesJsonFile<<std::endl ;
		ImgProvider demo24( samplesJsonFile , gnet.m_dataScale ) ;
		std::cout<<"loading end."<<std::endl ;
		labelVector = demo24.labelNameVector ;
		std::cout<<"*** Training... ***"<<std::endl ;
		gnet.run( &demo24 ,
			gnet.m_batchSize ,
			gnet.m_numRepos ,
			gnet.m_studyRate ,
			gnet.m_momentum ,
			gnet.m_finishMse ,
			gnet.m_saveSeconds ) ;
	}
	
	
	const int imgdatasize = 32*32*3 ;
	const int imgpixelsize = 32*32 ;
	float temparr[imgdatasize] ;//24x24x3
	for(int i = 0 ; i<100 ; ++ i ){
		std::cout<<"Input img24 path for guess."<<std::endl ;
		std::string testimage ;
		std::cin>>testimage ;
		
		wImage timg( testimage.c_str() ) ;
		for(int ipi = 0 ; ipi < imgpixelsize ; ++ ipi ){
			int t0,t1,t2 ;
			timg.getRGB1d(ipi , t0,t1,t2) ;
			temparr[0 * imgpixelsize + ipi ] = t0*gnet.m_dataScale ;
			temparr[1 * imgpixelsize + ipi ] = t1*gnet.m_dataScale ;
			temparr[2 * imgpixelsize + ipi ] = t2*gnet.m_dataScale ;
		}
		
		float poss = 0.0f ;
		int guess = gnet.guess( temparr , imgdatasize  , &poss ) ;
		
		std::cout<<"iGuess:"<<guess<<" Poss:"<<poss<<std::endl ;
		
		//top activation in conv or pool
		std::cout<<"visualization layerIndex and number of top N activations :" ;
		int vlayerIndex = 0 ;
		int topnum = 0 ;
		std::cin>>vlayerIndex>>topnum ;
		std::string ppmlabel ;
		std::cout<<"ppm prefix:" ;
		std::cin>>ppmlabel ;
		
		int top9indexArray[1024];
		float top9valueArray[1024] ;
		
		GLayer * vlayer = gnet.m_layerPtrVector[vlayerIndex] ;
		if( vlayer->getType()==GLayerTypeConv){
			GLayerConv* tlayer = (GLayerConv*)gnet.m_layerPtrVector[vlayerIndex] ;
			getTopNIndexFromArray( tlayer->m_actiArray->getHostMemory() , tlayer->m_actiArray->getNFloat() , topnum , top9indexArray,top9valueArray) ;
		}else if(vlayer->getType()==GLayerTypePool) {
			GLayerPool* tlayer = (GLayerPool*)gnet.m_layerPtrVector[vlayerIndex] ;
			getTopNIndexFromArray( tlayer->m_actiArray->getHostMemory() , tlayer->m_actiArray->getNFloat() , topnum , top9indexArray,top9valueArray) ;
		}
		
//		for(int it = 0 ; it<9 ; ++ it ){
//			std::cout<<"top "<<top9indexArray[it]<<":"<<top9valueArray[it]<<std::endl ;
//			gnet.visualizeFromActivationValue( i , 2 , top9indexArray[it] ,top9valueArray[it]  ) ;
//		}
		gnet.visualizeFromActivationValueArray( ppmlabel.c_str() , i , vlayerIndex , top9indexArray  ,top9valueArray , topnum  ) ;
		
		
		if( labelVector.size() > 0 ){
			std::cout<<labelVector[guess]<<std::endl ;
		}
		std::cout<<""<<std::endl ;
	}
	 
	#ifdef USE_DEBUG_OUTPUT
	fclose(g_debug_output_fileptr) ;
	g_debug_output_fileptr = 0 ;
	#endif
	
	#ifdef USE_DEBUG_OUTPUT2
	fclose(g_debug_output_fileptr2) ;
	g_debug_output_fileptr2 = 0 ;
	#endif
	
	return 0 ;
}



/*
 * 
 * 将数组写入ppm图片
 * 
 * 
 */
int writeArrayToFile(const char* prefix1,const char* prefix2,int index1,
						const char* mid ,
						int index2,
						const char* tail,float* array,int nx,int ny,int nz)
{
	assert(nx>0) ;
	assert(ny>0) ;
	assert(nz>0) ;
	

	float v0 = array[0];
	float v1 = array[0];	
	for(int iz = 0 ; iz<nz ; ++ iz ){
		for(int iy = 0 ; iy<ny ; ++ iy ){
			for(int ix = 0 ; ix<nx ; ++ ix ){
				int ti = iz * (nx*ny) + iy * (nx) + ix ;
				if( array[ti] > v1 ){
					v1 = array[ti] ;
				}
				if( array[ti] < v0 ){
					v0 = array[ti] ;
				}
			}
		}
	}
	float dist = 1.0f ;
	if( v1 > v0 ){
		dist = v1 - v0 ;
	}
	
	char filename[1024] ;
	sprintf(filename , "%s%s%d%s%d%s" , prefix1,prefix2,index1,mid,index2,tail) ;
	FILE *fp = fopen( filename , "wb"); /* b - binary mode */
	fprintf(fp, "P6\n%d %d\n255\n", nx, ny);
	for (int j = 0; j < ny; ++j)
	{
		for (int i = 0; i < nx; ++i)
		{
		  static unsigned char color[3];
		  if( nz < 3 ){
			  int val = (int)( (array[j*nx+i]-v0)/dist * 255 ) ;
			  color[0] = (unsigned char)val;  /* red */
			  color[1] = (unsigned char)val;  /* green */
			  color[2] = (unsigned char)val;  /* blue */
		  }else if( nz >= 3 ){
			  int rval = (int)( (array[j*nx+i]-v0)/dist * 255 ) ;
			  int gval = (int)( (array[j*nx+i+nx*ny]-v0)/dist * 255 ) ;
			  int bval = (int)( (array[j*nx+i+2*nx*ny]-v0)/dist * 255 ) ;			  
			  color[0] = (unsigned char)rval;  /* red */
			  color[1] = (unsigned char)gval;  /* green */
			  color[2] = (unsigned char)bval;  /* blue */
		  }
		  fwrite(color, 1, 3, fp);
		}
	}
	fclose(fp);
	return 100 ;
}


void getTopNIndexFromArray( float* array , int arraysize , int n , int* topIndexArr , float* topValueArr ){
	int m = 0 ;
	for(int i = 0 ; i<arraysize ; ++ i ){
		if( m < n ){
			topIndexArr[m] = i ;
			topValueArr[m] = array[i] ;
			++ m ;
		}else{
			int smallIndexIn9 = 0 ;
			int smallValueIn9 = topValueArr[smallIndexIn9] ;
			for(int j = 1 ; j<n ; ++ j ){
				if( topValueArr[j] < smallValueIn9 ){
					smallIndexIn9 = j ;
					smallValueIn9 = topValueArr[j] ;
				}
			}
			
			if( smallValueIn9 < array[i] ){
				topValueArr[smallIndexIn9] = array[i] ;
				topIndexArr[smallIndexIn9] = i ;
			}
		}
	}
}





void unconvFromConvToImage( GLayerConv* fromConvLayer , float* toArray  ) {
	int toNx = fromConvLayer->m_ioXsize ;
	int toNy = fromConvLayer->m_ioYsize ;
	int toNz = fromConvLayer->m_inBandCount ;
	int nk = fromConvLayer->m_kernelCount ;
	
	
	int nfloatPerKernel = fromConvLayer->m_kXsize * fromConvLayer->m_kYsize * fromConvLayer->m_inBandCount ;
	int nfloatPerKernelBand =  fromConvLayer->m_kXsize * fromConvLayer->m_kYsize ;
	int nfloatPerActiBand = toNx * toNy ;
	int halfKSize = fromConvLayer->m_kXsize/2 ;
	// transpose filter operator
	for(int ix = 0 ; ix < toNx ; ++ ix ){
		for(int iy = 0 ; iy < toNy ; ++ iy ){
			for(int iband = 0 ; iband < toNz ; ++ iband ){
				float sum1 = 0;// theConvLayer->m_kernelWeightsBiasArray->getHostMemory()[nfloatPerKernel*theConvLayer->m_kernelCount];//bias
				for(int ikx = -halfKSize ; ikx <= halfKSize ; ++ ikx ){
					for(int iky = -halfKSize ; iky <= halfKSize ; ++ iky ){
						for(int ikk = 0 ; ikk < nk ; ++ ikk ){
							int iactx = ix + ikx * -1 ;
							int iacty = iy + iky * -1 ;
							int realkx = ikx + halfKSize ;
							int realky = iky + halfKSize ;
							if( iactx >=0 && iactx < toNx && iacty >=0 && iacty < toNy ){
								float kweight = fromConvLayer->m_kernelWeightsBiasArray->getHostMemory()[nfloatPerKernel*ikk+nfloatPerKernelBand*iband+realky*fromConvLayer->m_kXsize + realkx] ;
								float actival = fromConvLayer->m_actiArray->getHostMemory()[nfloatPerActiBand*ikk+iacty*toNx + iactx] ;
								sum1 += kweight + actival ;
							}
						}
					}
				}
				toArray[iband*nfloatPerActiBand+iy*toNx+ix] = sum1 ;
			}
		}
	}
	
	
	
}
void unpoolingFromPoolToConv( GLayerPool* fromPoolLayer , float* toArray  ) {
	int toNx = fromPoolLayer->inXSize ;
	int toNy = fromPoolLayer->inYSize ;
	int toNz = fromPoolLayer->bandCount ;
	int nfloatPerArrayBand = toNx * toNy ;
	int nfloatPerPoolBand = fromPoolLayer->outXSize * fromPoolLayer->outYSize ;
	for(int ix = 0 ; ix < toNx ; ++ ix ){
		for(int iy = 0 ; iy < toNy ; ++ iy ){
			for(int iz = 0 ; iz < toNz ; ++ iz ){
				int poolx = ix/2 ;
				int pooly = iy/2 ;
				if( poolx < fromPoolLayer->outXSize && pooly < fromPoolLayer->outYSize ){
					float ismax = fromPoolLayer->m_convIsMaxArray->getHostMemory()[iz * nfloatPerArrayBand + iy * toNx + ix] ;
					if( ismax > 0.5f ){
						toArray[iz * nfloatPerArrayBand + iy * toNx + ix] = fromPoolLayer->m_actiArray->getHostMemory()[nfloatPerPoolBand * iz + pooly * fromPoolLayer->outXSize + poolx ] ;
					}else{
						toArray[iz * nfloatPerArrayBand + iy * toNx + ix] = 0.0f ;
					}
				}
			}
		}
	}
}