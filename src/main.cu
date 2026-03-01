#include "ImageBase.h"
#include <stdio.h>
#include <iostream>
#include <fstream> 
#include <vector> 
#include <string> 
#include <cstring> 
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

__global__
void sumImages(unsigned char* d_in, unsigned int* d_out, int width, int height) {
    __shared__ unsigned int sharedSum[16][16];//shared with the rest of the block

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int z = blockIdx.z;

    if (x < width && y < height)
    {
        int imgSize = width * height;
        int index = z * imgSize + y * width + x;


        sharedSum[ty][tx] = d_in[index];
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        {
            if (tx < stride)
            {
                sharedSum[ty][tx] += sharedSum[ty][tx + stride];
            }
            __syncthreads();
        }

        if (tx == 0)
        {
            for (int stride = blockDim.y / 2; stride > 0; stride /= 2)
            {
                if (ty < stride)
                {
                    sharedSum[ty][0] += sharedSum[ty+stride][0];
                }
                __syncthreads();
            }

            if(ty == 0){
                atomicAdd(&d_out[z], sharedSum[0][0]);
            }
        }
    }
}

__global__
void areaSums(unsigned char* d_in, unsigned int* d_out, int width, int height, int divider) {
    __shared__ unsigned int sharedSum[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int z = blockIdx.z;
    if (x < width && y < height)
    {
        
        int imgSize = width * height;
        int index = z * imgSize + y * width + x;

        sharedSum[ty][tx] = d_in[index];
        __syncthreads();

        int areaWidth = width/divider;
        int areaHeight = height/divider;
        int areaX = x/areaWidth;
        int areaY = y/areaHeight;


        for (int stride = 1; stride < min(blockDim.x, areaWidth); stride *= 2)
        {
            if (tx%(stride*2) == 0)
            {
                sharedSum[ty][tx] += sharedSum[ty][tx + stride];
            }
            __syncthreads();
        }
        

        if(tx%areaWidth == 0){
            for (int stride = 1; stride < min(blockDim.y, areaHeight); stride *= 2)
            {
                if (ty%(stride*2) == 0)
                {
                    sharedSum[ty][tx] += sharedSum[ty + stride][tx];
                }
                __syncthreads();
            }

            if(ty%areaHeight == 0){
                atomicAdd(&d_out[divider*divider*z + areaY * divider + areaX], sharedSum[ty][tx]);
            }
        } 
    }
}

__global__
void division(unsigned int* d_in, unsigned char* d_out, int N, int divider) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < N) d_out[x] = d_in[x]/divider;
}

std::vector<unsigned char> getImagesLocalMeans(std::vector<unsigned char> imageChar, int width, int height, int divider){
    size_t imgSize = width * height;
    size_t localSize = (width/divider) * (height/divider);
    size_t areaNbr = divider*divider;

    int imgNbr = imageChar.size()/(width*height);


    int bSize = 16;
    dim3 sumBlockSize(bSize, bSize);
    dim3 sumGridSize((width + bSize-1) / bSize, (height + bSize-1) / bSize, imgNbr);

    dim3 meanBlockSize(bSize*bSize);
    dim3 meanGridSize(imgNbr * areaNbr);

    unsigned char *d_in;
    cudaMalloc((void**)&d_in, imgNbr * imgSize * sizeof(unsigned char));
    unsigned int *d_out_tot;
    cudaMalloc((void**)&d_out_tot, imgNbr * areaNbr * sizeof(unsigned int));
    unsigned char *d_out_mean;
    cudaMalloc((void**)&d_out_mean, imgNbr * areaNbr * sizeof(unsigned char));

    cudaMemcpy(d_in, imageChar.data(), imgNbr * imgSize* sizeof(unsigned char), cudaMemcpyHostToDevice);


    areaSums<<<sumGridSize, sumBlockSize>>>(d_in, d_out_tot, width, height, divider);
    cudaDeviceSynchronize();

    division<<<meanGridSize, meanBlockSize>>>(d_out_tot, d_out_mean, imgNbr * areaNbr, localSize);
    cudaDeviceSynchronize();

    std::vector<unsigned char> hostOut(imgNbr * areaNbr);
    cudaMemcpy(hostOut.data(), d_out_mean, imgNbr * areaNbr * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    std::vector<unsigned int> testData(imgNbr * areaNbr);
    cudaMemcpy(testData.data(), d_out_tot, imgNbr * areaNbr * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out_tot);
    cudaFree(d_out_mean);
    return hostOut;
}

std::vector<unsigned char> getImagesMeans(std::vector<unsigned char> imagesChar, int width, int height){
    size_t imgNbr = imagesChar.size()/(width*height);
    size_t imgSize = width * height;
    size_t totalSize = imgSize * imgNbr;

    int bSize = 16;
    dim3 sumBlockSize(bSize, bSize);
    dim3 sumGridSize((width + bSize-1) / bSize, (height + bSize-1) / bSize, imgNbr);

    dim3 meanBlockSize(bSize*bSize);
    dim3 meanGridSize((imgNbr + bSize*bSize-1) / (bSize*bSize));

    unsigned char *d_in;
    cudaMalloc((void**)&d_in, totalSize* sizeof(unsigned char));
    unsigned int *d_out_tot;
    cudaMalloc((void**)&d_out_tot, imgNbr * sizeof(unsigned int));
    unsigned char *d_out_mean;
    cudaMalloc((void**)&d_out_mean, imgNbr * sizeof(unsigned char));

    cudaMemcpy(d_in, imagesChar.data(), totalSize* sizeof(unsigned char), cudaMemcpyHostToDevice);


    sumImages<<<sumGridSize, sumBlockSize>>>(d_in, d_out_tot, width, height);
    cudaDeviceSynchronize();

    division<<<meanGridSize, meanBlockSize>>>(d_out_tot, d_out_mean, imgNbr, width*height);
    cudaDeviceSynchronize();


    std::vector<unsigned char> hostOut(imgNbr);
    cudaMemcpy(hostOut.data(), d_out_mean, imgNbr * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out_tot);
    cudaFree(d_out_mean);
    return hostOut;
}

ImageBase composeImg(const std::vector<unsigned char> &imagesChar, int imgNbr, const std::vector<int> &composition){
    int smallImgSize = imagesChar.size()/imgNbr;
    int smallImgSide = sqrt(smallImgSize);
    int sideSize = sqrt(composition.size());

    ImageBase imOut = ImageBase(sideSize*smallImgSide, sideSize*smallImgSide, false);

    for(int offX = 0; offX < sideSize; offX++)
        for(int offY = 0; offY < sideSize; offY++){
            for(int x = 0; x < smallImgSide; x ++){
                for(int y = 0; y < smallImgSide; y++){
                    imOut[x+(offX*smallImgSide)][y+(offY*smallImgSide)] = imagesChar[composition[offX*sideSize + offY]*smallImgSize + x*smallImgSide + y];
                }
            }
        }
    return imOut;
}

std::vector<int> orderImg(const std::vector<unsigned char> &imIn, const std::vector<unsigned char> &imMeans){
    std::vector<int> outValues = std::vector<int>();
    for(int i = 0; i < imIn.size(); i ++){
        int bestIndex = 0;
        for(int j = 1; j < imMeans.size(); j ++){
            if((imIn[i]-imMeans[j]) * (imIn[i]-imMeans[j]) <  (imIn[i]-imMeans[bestIndex])*(imIn[i]-imMeans[bestIndex])){
                bestIndex = j;
            }
        }
        outValues.push_back(bestIndex);
    }
    return outValues;
}

int main(int argc, char **argv)
{

    //attribution de l'image d'entrée
    char cImageIn[250] = "08.pgm";
    ImageBase* imIn = new ImageBase();
    imIn->load(cImageIn);
    std::vector<unsigned char> imInChar;
    unsigned char* imInData = imIn->getData();
    imInChar.insert(imInChar.end(), imInData, imInData + imIn->getWidth() * imIn->getHeight());

    //attribution de la liste d'image de traitement
    char folderPath[250] = "Images/";
    std::vector<ImageBase*> images= std::vector<ImageBase*>();
    std::vector<unsigned char> imagesChar;
    int width = 512;
    int height = 512;
    int imgNbr = 0;
    for (const auto &entry : fs::directory_iterator(folderPath))
    {
        if (entry.is_regular_file())
        {
            std::string path = entry.path().string();
            if (entry.path().extension() == ".pgm")
            {
                ImageBase* img = new ImageBase();
                img->load(path.c_str());
                if(img->getHeight() == height && img->getWidth() == width){
                    images.push_back(img);
                    unsigned char* data = img->getData();
                    imagesChar.insert(imagesChar.end(), data, data + width * height);
                    imgNbr++;
                }
                std::cout << "Loaded: " << path << std::endl;
            }
        }
    }

    int sideOfImage = 256;
    int sideOfSmallImagesInPixel = 64;
    std::vector<unsigned char> imgsMeans = getImagesMeans(imagesChar, width, height);
    std::vector<unsigned char> imInLocalMeans = getImagesLocalMeans(imInChar, width, height, sideOfImage);
    std::vector<unsigned char> imgsLocalMeans = getImagesLocalMeans(imagesChar, width, height, sideOfSmallImagesInPixel);
    std::vector<int> order = orderImg(imInLocalMeans, imgsMeans);
    // ImageBase imOut = ImageBase(width, height, false);
    // for(int x = 0; x < width; x ++){
    //     for(int y = 0; y < height; y++){
    //         int meanX = x/(height/divider);
    //         int meanY = y/(width/divider);
    //         int mean = imInLocalMeans[meanX*divider+meanY];
    //         imOut[x][y] = mean;
    //     }
    // }
    ImageBase imOut = composeImg(imgsLocalMeans, images.size(), order);
    
    char cImageOut[250] = "out.pgm";
    imOut.save(cImageOut);
    return 0;
}