#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPIFFS.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"

const char* ssid = "Fatemeh";
const char* password = "Mohammad";
const int serverPort = 6000;
const int MODEL_INPUT_SIZE = 2048; // model's input size

WiFiUDP udp;

namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    constexpr int kTensorArenaSize = 30 * 1024; 
    byte tensor_arena[kTensorArenaSize];
}

char packetBuffer[16 * 1024];
float input_data[MODEL_INPUT_SIZE];

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    Serial.print("Microcontroller IP address: ");
    Serial.println(WiFi.localIP());

    // Initialize SPIFFS
    if (!SPIFFS.begin(true)) {
        Serial.println("An error has occurred while mounting SPIFFS");
        return;
    }

    // Open and read the model file
    File modelFile = SPIFFS.open("/model.tflite", "r");
    if (!modelFile) {
        Serial.println("Failed to open model file");
        return;
    }

    size_t modelSize = modelFile.size();
    byte* modelBuffer = (byte*)malloc(modelSize);
    if (!modelBuffer) {
        Serial.println("Failed to allocate memory for model");
        modelFile.close();
        return;
    }
    modelFile.read(modelBuffer, modelSize);
    modelFile.close();

//  Initialize TensorFlow Lite
    tflite::MicroErrorReporter microErrorReporter;
    error_reporter = &microErrorReporter;
    const tflite::Model* model = tflite::GetModel(modelBuffer);
    tflite::AllOpsResolver resolver;
    tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors");
        free(modelBuffer);
        return;
    }
    // // At this point, the model is loaded into the interpreter, and the model buffer is no longer needed.
    // free(modelBuffer); // Free the model buffer to reclaim memory

    input = interpreter->input(0);
    output = interpreter->output(0);


    udp.begin(serverPort);
    Serial.println("UDP Server started, waiting for messages...");
}

void loop() {
    int packetSize = udp.parsePacket();
    if (packetSize) {
        int len = udp.read(packetBuffer, sizeof(packetBuffer));
        if (len > 0) {
            packetBuffer[len] = '\0';
        }
        Serial.print("Received packet: ");
        Serial.println(packetBuffer);
        // Parse the received packet and update the input_data
        char* token = strtok(packetBuffer, ",");
        int i = 0;
        while (token != NULL && i < MODEL_INPUT_SIZE) {
            input_data[i++] = atof(token);
            token = strtok(NULL, ",");
        }
        
        // Update model input tensor
        for (i = 0; i < MODEL_INPUT_SIZE; i++) {
            input->data.f[i] = input_data[i];
        }

        unsigned int freeMemoryBefore = heap_caps_get_free_size(MALLOC_CAP_8BIT);

        Serial.print("Memory before: ");
        Serial.println(freeMemoryBefore);

        unsigned long startTime = micros();
        
        interpreter->Invoke();
        unsigned long endTime = micros();

        int numberOfClasses = output->dims->data[output->dims->size - 1];
        float* output_data = output->data.f;
        Serial.print("Output probabilities: ");
        for (int i = 0; i < numberOfClasses; ++i) {
            Serial.print(output_data[i]);
            Serial.print(" ");
        }
        Serial.println();

        int maxIndex = 0;
        for (int i = 1; i < numberOfClasses; ++i) {
            if (output_data[i] > output_data[maxIndex]) {
                maxIndex = i;
            }
        }

        unsigned long inferenceTime = endTime - startTime;
        unsigned int freeMemoryAfter = heap_caps_get_free_size(MALLOC_CAP_8BIT);

        Serial.print("Memory after: ");
        Serial.println(freeMemoryAfter);


        unsigned int artificialMemoryChange = freeMemoryBefore - freeMemoryAfter;
        


        Serial.print("Memory used: ");
        Serial.print(artificialMemoryChange);
        Serial.println(" bytes");

        Serial.print("Inference time: ");
        Serial.print(inferenceTime);
        Serial.println(" microseconds");

        Serial.print("Predicted class index: ");
        Serial.println(maxIndex);

        Serial.println("Prediction done.");

        float results[3] = {static_cast<float>(artificialMemoryChange), static_cast<float>(inferenceTime), static_cast<float>(maxIndex)};
        udp.beginPacket(udp.remoteIP(), udp.remotePort());
        udp.write((uint8_t*)results, sizeof(results));
        udp.endPacket();

        Serial.print("Data sent back to client: Memory: ");
        Serial.print(results[0]);
        Serial.print(", Time: ");
        Serial.print(results[1]);
        Serial.print(", Class Index: ");
        Serial.println(results[2]);
    }
    delay(10); // Short delay before the next loop iteration
}



