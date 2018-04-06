#include <iostream>
#include <fcntl.h>

#include <math.h>
#include <iomanip>
#include <fstream>
#include <ctime>

#include <visp/vpDebug.h>
#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpImageSimulator.h>
#include <visp/vpDisplayX.h>

#include <string>     // std::string, std::to_string

#include <visp3/gui/vpPlot.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpImageTools.h>

#include <visp3/core/vpExponentialMap.h>
//#include <Python.h>
#include <stdlib.h>

//new header files
#include <stdio.h>
#include <python2.7/Python.h>

//opencv

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std ;

void
computeError3D(vpHomogeneousMatrix &cdTc, vpColVector &cdrc)
{
    vpPoseVector _cdrc(cdTc) ;
    cdrc = (vpColVector)_cdrc ;
}

void
computeInteractionMatrix3D(vpHomogeneousMatrix &cdTc,  vpMatrix &Lx)
{

    vpRotationMatrix cdRc(cdTc) ;
    vpThetaUVector tu(cdTc) ;

    vpColVector u ;
    double theta ;

    tu.extract(theta,u);
    vpMatrix Lw(3,3) ;
    Lw[0][0] = 1 ;
    Lw[1][1] = 1 ;
    Lw[2][2] = 1 ;

    vpMatrix sku = vpColVector::skew(u) ;
    Lw += (theta/2.0)*sku ;
    Lw += (1-vpMath::sinc(theta)/vpMath::sqr(vpMath::sinc(theta/2.0)))*sku*sku ;

    Lx.resize(6,6) ;
    Lx = 0 ;
    for (int i=0 ; i < 3 ; i++){   // bloc translation
      for (int j=0 ; j < 3 ; j++)
      {
          Lx[i][j] = cdRc[i][j] ;
          Lx[i+3][j+3] = Lw[i][j] ;
      }
    }    




}


 int bytesToInt(unsigned char* b, unsigned length){
  int val = 0;
  int j = 0;
  for (int i = length-1; i >= 0; --i) {
    val += (b[i] & 0xFF) << (8*j);
    ++j;
    }
    return val;
  }
  
  
  //-----END of bytesToInt------------------------------------------------------------
  vector<float> bytesToFloatArray(unsigned char* b, unsigned length, unsigned arrayLength){
    typedef union {
      unsigned char b[4];
      float f;
    } bfloat;
    bfloat tmp;

    //float val[arrayLength];
    vector<float> val(arrayLength);
    //cout << "Length/arrayLength = " << length <<"/" <<arrayLength << endl;
    for(int j = 0 ; j < arrayLength ; j++){
      for(int i = 0 ; i < length ; i++) { // treat only the first value
        tmp.b[i] = b[i+j*length];
      }
      //cout << "tmp = " << tmp.f << endl;
      val[j] = tmp.f;
      //cout << "val[j] = " << val[j] << endl;
    }
    return val;
  }
    
  //--------End of bytesToFloatArray----------------------------------------------------------------------------------
  
  
  
  vector<double> queryServer(vpImage<unsigned char> &image){
  
 
  FILE *file1;
  int fifo_server,fifo_client;
  char *str;
  char *buf;
  int choice=1;
  int bufSize = 30;

  //printf("==== Starting Client process ====");

  unsigned char *bufInt;
  int result = 0;



  //cout << "Original image size = " << image.cols << "/" << image.rows << endl;
  int *testList ;
  testList = new int[image.getRows()*image.getCols()];
  for(int i = 0 ; i < image.cols ; i++){
    //cout << "i = " << i << "/" << image.cols << endl;
    for(int j = 0 ; j < image.rows ; j++){
    //cout << "j = " << j << "/" << image.rows << endl;
    testList[i*image.rows + j] = ((int)image[j][i]));
    //testList[i*image.rows + j] = 255;
    }
  }


  // Sending array size then array bulk
  //fifo_server=open("/dev/shm/fifo",O_RDWR);
  fifo_server=open("/dev/shm/fifo_server",O_RDWR);
  //fifo_server=open("/dev/shm/fifo_server",O_WRONLY);
  if(fifo_server < 0) {
  printf("Error in opening file");
  exit(-1);
  }
  int arraySize = (sizeof(testList)/sizeof(*testList));
  write(fifo_server, &arraySize,sizeof(int)); // We write an int array
  write(fifo_server,testList,arraySize*sizeof(int)); // We write an int array
  close(fifo_server);


  //fifo_client=open("/dev/shm/fifo",O_RDWR);
  fifo_client=open("/dev/shm/fifo_client",O_RDWR);
  if(fifo_client < 0) {
  printf("Error in opening file");
  exit(-1);
  }


  // Receiving array size  
  //unsigned char *bufInt;
  bufInt=(unsigned char*)malloc(1*sizeof(int));
  //printf("Size of int = %d", sizeof(int));

  read (fifo_client,bufInt,1*sizeof(int));
  
  result = bytesToInt(bufInt,4);
  //printf("\n *** Reply size from server is '%d' ***\n",result);

  // Receiving FLOAT array
  unsigned char *bufArray;
  bufArray = (unsigned char*)malloc(result*sizeof(float));

  read (fifo_client,bufArray,result*sizeof(float));

  vector<float> resultArray = bytesToFloatArray(bufArray, 4, result);
  close(fifo_client);

  //cout << "Descriptor Received = " << resultArray[0] << "/" << resultArray[1] << "/" << resultArray[2] << "/" << resultArray[3] << "/" << resultArray[4] << "/" << resultArray[5] << endl;
  //cout << "Descriptor Received 2 = " << resultArray[0] << "/" << resultArray[1] << "/" << resultArray[2] << "/" << resultArray[3] << "/" << resultArray[4] << "/" << resultArray[5] << endl;
  //for(int i = 0 ; i < result ; i++){
  //  cout << "desc[" << i << "] = " << resultArray[i] << endl;
  //}

  /*cout << "Sending com time = " << elapsedSend*1000 << "ms" << endl;
  cout << "Receiving com time = " << elapsedReceive*1000 << "ms" << endl;
  cout << "Debug time (int read) = " << elapsedDebug*1000 << "ms" << endl;
  cout << "Debug2 time (array read) = " << elapsedDebug2*1000 << "ms" << endl;
  cout << "Debug3 time (opening pipe) = " << elapsedDebug3*1000 << "ms" << endl;*/

  //cout << "Vector result = " << endl;
  vector<double> resultVector(6);
  for(int i = 0 ; i < 6 ; i++){
  if(i < 3){
    resultVector[i] = resultArray[i]/1000; // Scaling from [mm] to [m]
  }
  else{
    resultVector[i] = resultArray[i]; 
  }
  //cout << "====" << endl;
  //cout <<  resultArray[i] << endl;
  //cout <<  resultVector[i] << endl;
  }
  //cout << "DescriptorVector Received  = " << resultVector[0] << "/" << resultVector[1] << "/" << resultVector[2] << "/" << resultVector[3] << "/" << resultVector[4] << "/" << resultVector[5] << endl;
  
  delete [] testList ; //?? EM

  return resultVector;
  }


  //---------End of queryServer--------------------------------------------------------------------------------------------
  
  vpPoseVector getDirectionFromCNN(vpImage<unsigned char> &I){

  vector<double> result(6);
/*  vpImage<unsigned char> IErreur;
  IErreur.resize(im.getHeight(),im.getWidth());

  // Compute the difference image
  for(unsigned int j=0;j<IErreur.getWidth();j++)
    for(unsigned int  i=0;i<IErreur.getHeight();i++)
      IErreur[i][j]=128+(int)((int)im[i][j]-(int)desiredImage[i][j])/2;
*/
  // Convert it into 'Mat'
//Mat input;
//  vpImageConvert::convert(IErreur, input);
 // vpImageConvert::convert(I, input);
  result = queryServer(I);
  
  vpPoseVector r ;
  for (int i=0 ; i <6 ; i++) r[i] = result[i] ;
  
  // Return this pose estimate
  return r;
}
//-----------------End of getDirectionfromCNN------------------------------------------


int main()
{ 
    
  FILE* file;
  int argc;
  char * argv[3];


  vpTRACE("begin" ) ;

  vpPlot plot(4, 700, 700, 100, 200, "Curves...");


  char title[40];
  strncpy( title, "||e||", 40 );
  plot.setTitle(0,title);
  plot.initGraph(0,1);

  strncpy( title, "x-xd", 40 );
  plot.setTitle(1, title);
  plot.initGraph(1,6);

  strncpy( title, "camera velocity", 40 );
  plot.setTitle(2, title);
  plot.initGraph(2,6);


  strncpy( title, "Point position", 40 );
  plot.setTitle(3, title);
  plot.initGraph(3,6);


  cv::Mat mat_img; //initialize MAT variables

  int i,j;
  vpImage<unsigned char> I(224,224,0); //<unsigned char> for greyscale images
  vpImage<unsigned char> Id(224,224,0); //<unsigned char> for greyscale images
  vpImage<vpRGBa> Iimage(800,1200);


  vpImageIo::read(Iimage,"../data/hollywood-triangle.jpg") ;

  // Cette partie ne sert qu'a la simulation
  // on positionne un poster dans le repere Rw

  // This part is only for simulation
  // we position a poster in the rep Rw

  //  double L = 0.400 ;
  //  double l = 0.300;

  double L = 0.400 ;
  double l = 0.300;

  // Initialise the 3D coordinates of the Iimage corners
  vpColVector X[4];
  for (int i = 0; i < 4; i++) X[i].resize(3);
  // Top left corner
  X[0][0] = -L;
  X[0][1] = -l;
  X[0][2] = 0;

  // Top right corner
  X[1][0] = L;
  X[1][1] = -l;
  X[1][2] = 0;

  // Bottom right corner
  X[2][0] = L;
  X[2][1] = l;
  X[2][2] = 0;

  //Bottom left corner
  X[3][0] = -L;
  X[3][1] = l;
  X[3][2] = 0;



  vpImageSimulator sim;
  sim.setInterpolationType(vpImageSimulator::BILINEAR_INTERPOLATION);

  sim.init(Iimage, X);

  // On définit une camera avec certain parametre u0 = 200, v0 = 150; px = py = 800
  //vpCameraParameters cam(1110.0, 1110.0, 333, 227);
  //old parameters
  vpCameraParameters cam(800.0, 800.0, 200, 150);
  cam.printParameters() ;

  vpMatrix K = cam.get_K() ;

  //Matrix of intrinsic parameters 

  cout << "Matrice des paramètres intrinsèques" << endl ;
  cout << K << endl ;

  /*
  // On positionne une camera c1 à la position c1Tw (ici le repere repère Rw est 2m devant Rc1 
  //We position a camera c1 at position c1Tw (here the reference mark Rw is 2m in front of Rc1
  vpHomogeneousMatrix  c1Tw(0,0,2.5,  vpMath::rad(0),vpMath::rad(0),0) ;
  //on simule l'image vue par c1 //we simulate the image seen by c1
  sim.setCameraPosition(c1Tw);
  // on recupère l'image //we recover the image
  sim.getImage(I1,cam);
  cout << "Image I1g " <<endl ;
  cout << c1Tw << endl ;
  */

  //desired position
  vpHomogeneousMatrix cdTw(0,0,1, vpMath::rad(0),vpMath::rad(0),0) ;
  sim.setCameraPosition(cdTw);
  sim.setCleanPreviousImage(true, vpColor::black); //set color, default is black
  sim.getImage(Id,cam);


  //current position

  // vpHomogeneousMatrix cTw(0.1,0,1, vpMath::rad(0),vpMath::rad(0),0) ;
  vpHomogeneousMatrix cTw(0.05,-0.02,1, vpMath::rad(10),vpMath::rad(-5),vpMath::rad(10)) ;
  sim.setCameraPosition(cTw);
  sim.setCleanPreviousImage(true, vpColor::black); //set color, default is black
  // on recupère l'image I2 //we recover the image I2
  sim.getImage(I,cam);

  vpImage<unsigned char> Idiff;
  Idiff = I;
  vpImageTools::imageDifference(I, Id, Idiff);

  // On affiche l'image I1 //We display image Id
  vpDisplayX dd(Idiff,10,10,"I-I*") ;
  vpDisplay::display(Idiff) ;
  vpDisplay::flush(Idiff) ;



  // Display current image
  vpDisplayX d(I,10,400,"I") ;
  vpDisplay::display(I) ;
  vpDisplay::flush(I) ;



  vpColVector e(6) ; //
  e[0] = 1 ; // moche mais sert juste a entrer dans la boucle...

  vpMatrix Lx ;

  vpColVector v ;
  double lambda = 0.1 ;
  int iter = 0 ;
  while (fabs(e.sumSquare()) > 1e-8)
  {

    sim.setCameraPosition(cTw);
    sim.setCleanPreviousImage(true, vpColor::black); //set color, default is black
    // on recupère l'image I2 //we recover the image I2
    sim.getImage(I,cam);

    vpDisplay::display(I) ;
    vpDisplay::flush(I) ;

    vpImageTools::imageDifference(I, Id, Idiff);
    // On affiche l'image I1 //We display image Id

    vpDisplay::display(Idiff) ;
    vpDisplay::flush(Idiff) ;


    // Here cdTc is obtain thanks to a simulation process
    vpHomogeneousMatrix cdTc ;
    //cdTc = cdTw*cTw.inverse() ;   // ----------------------------------------------------------cdTc here will be replaced by the one at (1) below. cdTc = CNN(I)
	

    //  return 0;

    // the two previous line should be be replace by something like
    vpPoseVector cdrc ;
    cdrc = getDirectionFromCNN(I) ;                              // ------------------------------------------> 1 
    cdTc.buildFrom(cdrc) ;

    // Calcul de l'erreur
    computeError3D(cdTc, e) ;
    // Calcul de la matrice d'interaction
    computeInteractionMatrix3D(cdTc, Lx) ;
    //        Calcul de la loi de commande
    vpMatrix Lp ;
    Lp = Lx.pseudoInverse() ;

    v = - lambda * Lp * e ;

    // Mis à jour de la position de la camera
    cTw = vpExponentialMap::direct(v).inverse()* cTw ;

    cout << "iter "<< iter <<" : "<< e.t() << endl ;

    iter++ ;

    //mis a jour de courbes
    vpPoseVector crw(cTw) ;
    plot.plot(0,0,iter, e.sumSquare()) ;
    plot.plot(1,iter, e) ;
    plot.plot(2,iter, v) ;
    plot.plot(3,iter,crw) ;


  }


  // sauvegarde des courbes
  plot.saveData(0,"e.txt","#");
  plot.saveData(1,"error.txt","#");
  plot.saveData(2,"v.txt","#");
  plot.saveData(3,"p.txt","#");

  int a ; cin >> a ;

  return 0;
}
