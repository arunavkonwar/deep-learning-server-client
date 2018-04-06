#include <iostream>
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
#include <Python.h>
#include <stdlib.h>


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
    for (int i=0 ; i < 3 ; i++)   // bloc translation
        for (int j=0 ; j < 3 ; j++)
        {
            Lx[i][j] = cdRc[i][j] ;
            Lx[i+3][j+3] = Lw[i][j] ;
        }


}


int main(int argc, char *argv[])
{


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


  MAT A,C; //initialize MAT variables
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
       cdTc = cdTw*cTw.inverse() ;   // ----------------------------------------------------------cdTc here will be replaced by the one at (1) below. cdTc = CNN(I)

		
	
	// Python embedder
	//Py_SetProgramName(argv[0]);  /* optional but recommended */
/*
	Py_Initialize();
	PyObject *obj = Py_BuildValue("s", "test.py");
 	FILE *file = _Py_fopen_obj(obj, "r+");
	if(file != NULL) {
		PyRun_SimpleFile(file, "test.py");
	}

*/

	//Py_SetProgramName(argv[0]);  /* optional but recommended */
	Py_Initialize();
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("print(1+2)");
	Py_Finalize();
//	return 0;

       // the two previous line should be be replace by something like
       // cdTc = CNN(I) ;                              // ------------------------------------------> 1 


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



   /*


    long k=0;
  // On positionne une camera c2 à la position c2Tw //Positioning a camera c2 at position c2Tw
  for(float i=-0.2;i<=0.2;i=i+0.01){
    for(float j=-0.1;j<=0.1;j=j+0.001){
      vpHomogeneousMatrix c2Tw(i,j,1,
        vpMath::rad(0),vpMath::rad(0),0) ; //0.1,0,2, vpMath::rad(0),vpMath::rad(0),0) ;
        //on simule l'image vue par c2 //we simulate the image seen by c2
        sim.setCameraPosition(c2Tw);
        sim.setCleanPreviousImage(true, vpColor::black); //set color, default is black
        // on recupère l'image I2 //we recover the image I2
        sim.getImage(I2,cam);
        cout << "Image I1d " <<endl ;
        cout << c2Tw << endl ;
        
        float io=floorf(i * 100) / 100;
        float jo=floorf(j * 100) / 100;

        //float io=(float)(((int)(i*10))/10.0);;
        //float jo=(float)(((int)(j*10))/10.0);

        string loli = to_string(io);
        string lolj = to_string(jo);  
        //cout<<fixed;
        //cout<<setprecision(2);
        k++;
        string lolk = to_string(k);
        //vpImageIo::write(I2,k+".jpg") ; //write to filename
        vpImageIo::write(I2,"generated_images/"+lolk + ".jpg");
        vpHomogeneousMatrix  c2Tc1 = c2Tw * c1Tw.inverse() ;
        vpPoseVector deltaT(c2Tc1) ; //  vector 6 (t,theta U)
        //cout << vpPoseVector << endl;
        cout<<c2Tc1[0];

        ofstream outfile;
        outfile.open("data.txt", ios_base::app);
        //outfile << loli+" "+lolj<<endl;
        outfile << deltaT.t() << endl;
        //cout << deltaT.t() << endl ;
        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout<<elapsed_secs<<endl;

    }  
  }
  */
 
 /*

  // On affiche l'image I1 //We display image I1
  vpDisplayX d1(I1,10,10,"I1") ;
  vpDisplay::display(I1) ;
  vpDisplay::flush(I1) ;

  

  // On affiche l'image I2 //We display image I2
  vpDisplayX d2(I2,10,400,"I2") ;
  vpDisplay::display(I2) ;
  vpDisplay::flush(I2) ;

  */


  // sauvegarde des images resultats (en jpg et ppm) //save images results (in jpg and ppm)
//  vpImageIo::write(I1,"I1.jpg") ;
//  vpImageIo::write(I1,"I1.pgm") ;

  //vpImageIo::write(I2,"I2.jpg") ;
  //vpImageIo::write(I2,"I2.pgm") ;

/*

  vpDisplay::getClick(I2) ;
  cout << "OK " << endl ;

  vpDisplay::close(I2) ;
  vpDisplay::close(I1) ;

 */ 


  return 0;
}
