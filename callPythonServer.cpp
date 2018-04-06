vector<double> getDirectionFromCNN(vpImage<unsigned char> im){
  vector<double> result(6);
  vpImage<unsigned char> IErreur;
  IErreur.resize(im.getHeight(),im.getWidth());

  // Compute the difference image
  for(unsigned int j=0;j<IErreur.getWidth();j++)
    for(unsigned int  i=0;i<IErreur.getHeight();i++)
      IErreur[i][j]=128+(int)((int)im[i][j]-(int)desiredImage[i][j])/2;

  // Convert it into 'Mat'
  Mat input;
  vpImageConvert::convert(IErreur, input);

  result = queryServer(input);
  // Return this pose estimate
  return result;
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

int* bytesToIntArray(unsigned char* b, unsigned length, unsigned arrayLength){
  int val[arrayLength];
  for(int i = 0 ; i < arrayLength ; i++){
  val[i] = 0;
  }
  for(int index = 0 ; index < arrayLength ; index++){
  int j = 0;
  int tmp = 0;
  for (int i = length-1; i >= 0; --i) {
  tmp += (b[index*length+i] & 0xFF) << (8*j);
  ++j;
  }
  val[index] = tmp;
  }
  return val;
}

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

vector<double> queryServer(Mat image){
FILE *file1;
int fifo_server,fifo_client;
char *str;
char *buf;
int choice=1;
int bufSize = 30;
// Starting time recording
struct timespec start, finish;
struct timespec startSend, finishSend;
struct timespec startReceive, finishReceive;
struct timespec startDebug, finishDebug;
struct timespec startDebug2, finishDebug2;
struct timespec startDebug3, finishDebug3;
double elapsed, elapsedSend, elapsedReceive, elapsedDebug, elapsedDebug2, elapsedDebug3;



//printf("==== Starting Client process ====");
clock_gettime(CLOCK_MONOTONIC, &start);
clock_gettime(CLOCK_MONOTONIC, &startSend);

unsigned char *bufInt;
int result = 0;

//printf("Client // Sending in array");
//Mat image;
//image = imread("im_0000000.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
//image = imread("cat_gray.jpg", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file


//namedWindow( "Display window Client", WINDOW_AUTOSIZE );// Create a window for display.
//imshow( "Display window Client", image );                   // Show our image inside it.
//waitKey(0);     


//cout << "Original image size = " << image.cols << "/" << image.rows << endl;
int testList[image.rows*image.cols];
for(int i = 0 ; i < image.cols ; i++){
  //cout << "i = " << i << "/" << image.cols << endl;
  for(int j = 0 ; j < image.rows ; j++){
  //cout << "j = " << j << "/" << image.rows << endl;
  testList[i*image.rows + j] = ((int)image.at<uchar>(j, i));
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
clock_gettime(CLOCK_MONOTONIC, &finishSend);

//printf("Receiving data (array size then body)");
clock_gettime(CLOCK_MONOTONIC, &startReceive);
clock_gettime(CLOCK_MONOTONIC, &startDebug3);
//fifo_client=open("/dev/shm/fifo",O_RDWR);
fifo_client=open("/dev/shm/fifo_client",O_RDWR);
if(fifo_client < 0) {
printf("Error in opening file");
exit(-1);
}
clock_gettime(CLOCK_MONOTONIC, &finishDebug3);

// Receiving array size  
//unsigned char *bufInt;
bufInt=(unsigned char*)malloc(1*sizeof(int));
//printf("Size of int = %d", sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &startDebug);
read (fifo_client,bufInt,1*sizeof(int));
clock_gettime(CLOCK_MONOTONIC, &finishDebug);

/*int */result = bytesToInt(bufInt,4);
//printf("\n *** Reply size from server is '%d' ***\n",result);

// Receiving FLOAT array
unsigned char *bufArray;
bufArray = (unsigned char*)malloc(result*sizeof(float));
clock_gettime(CLOCK_MONOTONIC, &startDebug2);
read (fifo_client,bufArray,result*sizeof(float));
clock_gettime(CLOCK_MONOTONIC, &finishDebug2);
vector<float> resultArray = bytesToFloatArray(bufArray, 4, result);
close(fifo_client);
clock_gettime(CLOCK_MONOTONIC, &finishReceive);
//cout << "Descriptor Received = " << resultArray[0] << "/" << resultArray[1] << "/" << resultArray[2] << "/" << resultArray[3] << "/" << resultArray[4] << "/" << resultArray[5] << endl;
//cout << "Descriptor Received 2 = " << resultArray[0] << "/" << resultArray[1] << "/" << resultArray[2] << "/" << resultArray[3] << "/" << resultArray[4] << "/" << resultArray[5] << endl;
//for(int i = 0 ; i < result ; i++){
//  cout << "desc[" << i << "] = " << resultArray[i] << endl;
//}

clock_gettime(CLOCK_MONOTONIC, &finish);
elapsed = (finish.tv_sec - start.tv_sec);
elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

elapsedSend = (finishSend.tv_sec - startSend.tv_sec);
elapsedSend += (finishSend.tv_nsec - startSend.tv_nsec) / 1000000000.0;

elapsedReceive = (finishReceive.tv_sec - startReceive.tv_sec);
elapsedReceive += (finishReceive.tv_nsec - startReceive.tv_nsec) / 1000000000.0;

elapsedDebug = (finishDebug.tv_sec - startDebug.tv_sec);
elapsedDebug += (finishDebug.tv_nsec - startDebug.tv_nsec) / 1000000000.0;

elapsedDebug2 = (finishDebug2.tv_sec - startDebug2.tv_sec);
elapsedDebug2 += (finishDebug2.tv_nsec - startDebug2.tv_nsec) / 1000000000.0;

elapsedDebug3 = (finishDebug3.tv_sec - startDebug3.tv_sec);
elapsedDebug3 += (finishDebug3.tv_nsec - startDebug3.tv_nsec) / 1000000000.0;

/*cout << "Sending com time = " << elapsedSend*1000 << "ms" << endl;
cout << "Receiving com time = " << elapsedReceive*1000 << "ms" << endl;
cout << "Debug time (int read) = " << elapsedDebug*1000 << "ms" << endl;
cout << "Debug2 time (array read) = " << elapsedDebug2*1000 << "ms" << endl;
cout << "Debug3 time (opening pipe) = " << elapsedDebug3*1000 << "ms" << endl;*/
cout << "Time elapsed Client Total = " << elapsed*1000 << "ms"<< endl;  

//cout << "Vector result = " << endl;
vector<double> resultVector(6);
for(int i = 0 ; i < 6 ; i++){
if(i < 3)
resultVector[i] = resultArray[i]/1000; // Scaling from [mm] to [m]
else
resultVector[i] = resultArray[i]; 
//cout << "====" << endl;
//cout <<  resultArray[i] << endl;
//cout <<  resultVector[i] << endl;
}
//cout << "DescriptorVector Received  = " << resultVector[0] << "/" << resultVector[1] << "/" << resultVector[2] << "/" << resultVector[3] << "/" << resultVector[4] << "/" << resultVector[5] << endl;

return resultVector;
}
