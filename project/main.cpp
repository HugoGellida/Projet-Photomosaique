#include "DatasetManager.h"
#include "ImageBase.h"
#include "ImageComposer.h"
#include "ImageEvaluator.h"
#include "ImageOrdering.h"
#include "ImageProcessor.h"
#include "Utils.h"
#include "VideoProcessor.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/Clipboard.hpp>

#include "Button.h"

// Video processing constants
const int WIDTH_INPUT = 512;
const int HEIGHT_INPUT = 512;
const int SIDE_OF_IMAGE = 64;
const int SMALL_TILE_SIZE = 32;
const int WIDTH_OUTPUT = SIDE_OF_IMAGE * SMALL_TILE_SIZE;
const int HEIGHT_OUTPUT = SIDE_OF_IMAGE * SMALL_TILE_SIZE;
const int FPS = 30;
const int WINDOW_WIDTH = 512;
const int WINDOW_HEIGHT = 512;
const  int REQUESTED_SIZE = 128;


void OpenSecondaryWindow(std::string inputFile, std::vector<unsigned char> &datasetData){
  bool isVideo = false;

  int imgNbr = datasetData.size()/(REQUESTED_SIZE*REQUESTED_SIZE);

  std::vector<unsigned char> datasetMeans =
      ImageProcessor::getGlobalMeans(datasetData, REQUESTED_SIZE, REQUESTED_SIZE);
  std::vector<unsigned char> datasetLocalMeans = ImageProcessor::getLocalMeans(
      datasetData, REQUESTED_SIZE, REQUESTED_SIZE, SMALL_TILE_SIZE);



  std::string inputPath = inputFile.c_str();
  std::string ext = inputPath.substr(inputPath.find_last_of(".") + 1);
  // ===========================
  // ===== FILE IS A VIDEO =====
  // ===========================
  if (ext == "mp4" || ext == "avi" || ext == "mov") {
    isVideo = true;
    //opening the video
    std::string ffmpegReadCmd = "ffmpeg -i " + inputPath +
                                " -f rawvideo -pix_fmt gray -s " +
                                std::to_string(WIDTH_OUTPUT) + "x" +
                                std::to_string(HEIGHT_OUTPUT) + " -r " +
                                std::to_string(FPS) + " -";

    FILE *ffmpegRead = popen(ffmpegReadCmd.c_str(), "r");
    if (!ffmpegRead) {
      std::cerr << "Failed to open ffmpeg input\n";
      return;
    }

    std::string ffmpegWriteCmd = "ffmpeg -y -f rawvideo -pix_fmt gray -s " +
                                 std::to_string(WIDTH_OUTPUT) + "x" +
                                 std::to_string(HEIGHT_OUTPUT) + " -r " +
                                 std::to_string(FPS) +
                                 " -i - -c:v libx264 -crf 28 Results/output.mp4";

    FILE *ffmpegWrite = popen(ffmpegWriteCmd.c_str(), "w");
    if (!ffmpegWrite) {
      std::cerr << "Failed to open ffmpeg output\n";
      pclose(ffmpegRead);
      return;
    }

    //processing the video
    size_t frameSize = WIDTH_OUTPUT * HEIGHT_OUTPUT;
    size_t frameInputSize = WIDTH_INPUT * HEIGHT_INPUT;
    ImageBase mosaic(WIDTH_OUTPUT, HEIGHT_OUTPUT, false);
    std::vector<unsigned char> buffer(frameInputSize);
    std::vector<int> prevComposition(SIDE_OF_IMAGE * SIDE_OF_IMAGE, -1);
    std::vector<bool> used(datasetMeans.size(), false);
    size_t frameIndex = 0;
    while (fread(buffer.data(), 1, frameInputSize, ffmpegRead) ==
           frameInputSize) {
      // Process frame
      ImageBase mosaicCPU = VideoProcessor::processFrameUnique(
          buffer.data(), WIDTH_INPUT, HEIGHT_INPUT,
          datasetLocalMeans, datasetMeans, prevComposition, SIDE_OF_IMAGE, used);

      //VideoProcessor::processFrameGPU(buffer.data(), WIDTH_INPUT, HEIGHT_INPUT, datasetLocalMeans, datasetMeans, prevComposition, mosaic, SIDE_OF_IMAGE);

      // Write to output
      size_t written = fwrite(mosaicCPU.getData(), 1, frameSize, ffmpegWrite);
      if (written != frameSize) {
        std::cerr << "\nWarning: incomplete write on frame " << frameIndex
                  << std::endl;
      }

      std::cout << "\rProcessing frame " << frameIndex++ << std::flush;
    }

    pclose(ffmpegRead);
    pclose(ffmpegWrite);

    std::cout << "\nDone. Video written to Results/output.mp4\n";

  } 
  // =============================
  // ===== FILE IS A pciture =====
  // =============================
  else { 
    ImageBase targetImage;
    if(targetImage.load(inputFile.c_str()) != 0){
      return;
    }

    auto targetLocalMeans = ImageProcessor::getLocalMeans(
        {targetImage.getData(),
         targetImage.getData() +
             targetImage.getWidth() * targetImage.getHeight()},
        targetImage.getWidth(), targetImage.getHeight(), SIDE_OF_IMAGE);

    auto compositionOrder =
        ImageOrdering::orderAllowRepeats(targetLocalMeans, datasetMeans);

    ImageBase result = ImageComposer::compose(
        datasetLocalMeans, imgNbr, compositionOrder);

    ImageBase resizedOrigin = Utils::resizeImage(
        {targetImage.getData(),
         targetImage.getData() +
             targetImage.getWidth() * targetImage.getHeight()},
        SIDE_OF_IMAGE, SMALL_TILE_SIZE);



    std::vector<sf::Uint8> pixels(WIDTH_OUTPUT * HEIGHT_OUTPUT * 4);

    for (int y = 0; y < HEIGHT_OUTPUT; y++) {
        for (int x = 0; x < WIDTH_OUTPUT; x++) {
            int i = (x + y * WIDTH_OUTPUT)*4;

            pixels[i + 0] = result[y][x];
            pixels[i + 1] = result[y][x];
            pixels[i + 2] = result[y][x];
            pixels[i + 3] = 255;
        }
    }

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "PhotoMosaique "+ (std::string)inputFile);
    sf::Texture texture;
    texture.create(WIDTH_OUTPUT, HEIGHT_OUTPUT);
    texture.update(pixels.data());
    sf::Sprite sprite(texture);
    sprite.setScale(WINDOW_WIDTH/(float)(WIDTH_OUTPUT), WINDOW_HEIGHT/(float)(HEIGHT_OUTPUT));

    while (window.isOpen()) {
          sf::Event event;
          while (
              window.pollEvent(event))
              if (event.type == sf::Event::Closed) window.close();
              if (event.type == sf::Event::TextEntered) {
                  if (event.text.unicode == 27) window.close();
              }
          window.clear();
          window.draw(sprite);
          window.display();
    }

    float PSNR = ImageEvaluator::PSNR(resizedOrigin, result);
    int diff = ImageEvaluator::diffHisto(resizedOrigin, result);

    std::cout << PSNR << " " << diff << std::endl;

    result.save("./Results/out.pgm");
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0]
              << " <input.pgm|input.mp4> <dataset_folder>\n";
    return 1;
  }

  // === Loader ===

  DatasetManager dataset(REQUESTED_SIZE, REQUESTED_SIZE);
  dataset.loadFromFolder(argv[2]);

  if (dataset.getImages().empty()) {
    std::cerr << "Error: Dataset is empty\n";
    return 1;
  }

  std::vector<unsigned char> datasetData = dataset.getConcatenatedData();

  bool isVideo = false;

  int imgNbr = datasetData.size()/(REQUESTED_SIZE*REQUESTED_SIZE);

  std::vector<unsigned char> datasetMeans =
      ImageProcessor::getGlobalMeans(datasetData, REQUESTED_SIZE, REQUESTED_SIZE);
  std::vector<unsigned char> datasetLocalMeans = ImageProcessor::getLocalMeans(
      datasetData, REQUESTED_SIZE, REQUESTED_SIZE, SMALL_TILE_SIZE);



  std::string inputPath = argv[1];
  std::string ext = inputPath.substr(inputPath.find_last_of(".") + 1);
  // ===========================
  // ===== FILE IS A VIDEO =====
  // ===========================
  if (ext == "mp4" || ext == "avi" || ext == "mov") {
    isVideo = true;
    //opening the video
    std::string ffmpegReadCmd = "ffmpeg -i " + inputPath +
                                " -f rawvideo -pix_fmt gray -s " +
                                std::to_string(WIDTH_OUTPUT) + "x" +
                                std::to_string(HEIGHT_OUTPUT) + " -r " +
                                std::to_string(FPS) + " -";

    FILE *ffmpegRead = popen(ffmpegReadCmd.c_str(), "r");
    if (!ffmpegRead) {
      std::cerr << "Failed to open ffmpeg input\n";
      return 0;
    }

    std::string ffmpegWriteCmd = "ffmpeg -y -f rawvideo -pix_fmt gray -s " +
                                 std::to_string(WIDTH_OUTPUT) + "x" +
                                 std::to_string(HEIGHT_OUTPUT) + " -r " +
                                 std::to_string(FPS) +
                                 " -i - -c:v libx264 -crf 28 Results/output.mp4";

    FILE *ffmpegWrite = popen(ffmpegWriteCmd.c_str(), "w");
    if (!ffmpegWrite) {
      std::cerr << "Failed to open ffmpeg output\n";
      pclose(ffmpegRead);
      return 0;
    }

    //processing the video
    size_t frameSize = WIDTH_OUTPUT * HEIGHT_OUTPUT;
    size_t frameInputSize = WIDTH_INPUT * HEIGHT_INPUT;
    ImageBase mosaic(WIDTH_OUTPUT, HEIGHT_OUTPUT, false);
    std::vector<unsigned char> buffer(frameInputSize);
    std::vector<int> prevComposition(SIDE_OF_IMAGE * SIDE_OF_IMAGE, -1);
    std::vector<bool> used(datasetMeans.size(), false);
    size_t frameIndex = 0;
    while (fread(buffer.data(), 1, frameInputSize, ffmpegRead) ==
           frameInputSize) {
      // Process frame
      ImageBase mosaicCPU = VideoProcessor::processFrameUnique(
          buffer.data(), WIDTH_INPUT, HEIGHT_INPUT,
          datasetLocalMeans, datasetMeans, prevComposition, SIDE_OF_IMAGE, used);

      //VideoProcessor::processFrameGPU(buffer.data(), WIDTH_INPUT, HEIGHT_INPUT, datasetLocalMeans, datasetMeans, prevComposition, mosaic, SIDE_OF_IMAGE);

      // Write to output
      size_t written = fwrite(mosaicCPU.getData(), 1, frameSize, ffmpegWrite);
      if (written != frameSize) {
        std::cerr << "\nWarning: incomplete write on frame " << frameIndex
                  << std::endl;
      }

      std::cout << "\rProcessing frame " << frameIndex++ << std::flush;
    }

    pclose(ffmpegRead);
    pclose(ffmpegWrite);

    std::cout << "\nDone. Video written to Results/output.mp4\n";

  } 


  // === Window ===
  sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "PhotoMosaique");

  sf::Font font;
  if (!font.loadFromFile("arial.ttf")) {
      return -1; 
  }

  sf::Text text;
  text.setFont(font);
  text.setCharacterSize(20);
  text.setFillColor(sf::Color::White);
  text.setPosition(20.f, 20.f);  

  std::string inputText = ""; 

  Button validateButton(20.f, 80.f, 200.f, 50.f, "Validate", font);

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) { window.close();}

        if (validateButton.isClicked(event, window)) {
            OpenSecondaryWindow(inputText, datasetData);
        }

        if (event.type == sf::Event::TextEntered) {
            if (event.text.unicode < 128) { 
                char enteredChar = static_cast<char>(event.text.unicode);
                if (enteredChar == 8 && !inputText.empty()) {
                    inputText.pop_back();
                }
                else if (enteredChar >= 32 && enteredChar <= 126) {
                    inputText += enteredChar;
                }
            }
            if (event.text.unicode == 13) {
              OpenSecondaryWindow(inputText, datasetData);
            }
            if (event.text.unicode == 3) {
                sf::Clipboard::setString(inputText);
            }
            if (event.text.unicode == 22) {
                inputText += sf::Clipboard::getString();
            }
        }
    }

    text.setString(inputText+"|");
    validateButton.updateHover(window);
    window.clear();
    validateButton.draw(window);
    window.draw(text);
    window.display();
  }
}