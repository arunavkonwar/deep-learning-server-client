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


using namespace std ;



int main()
{
  clock_t begin = clock();
  int i,j;
  //vpImage<unsigned char> I1(300,400,0);
  vpImage<unsigned char> I2(224,224,0); //<unsigned char> for greyscale images
  //vpImage<vpRGBa> I2(300,400,0); // <vpRGBa> for color images
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
   vpHomogeneousMatrix c1Tw(0,0,1,
        vpMath::rad(0),vpMath::rad(0),0) ; //0.1,0,2, vpMath::rad(0),vpMath::rad(0),0) ;
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


  
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout<<elapsed_secs<<endl;
  return 0;
}
