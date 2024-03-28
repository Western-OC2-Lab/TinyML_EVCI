#include <iostream>
#include <fstream>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

const int PORT = 6000;
const char* SERVER_IP = "192.168.255.188"; // IP address of the microcontroller
const int BUFFER_SIZE = 12; // Size for receiving 3 float values (3 * 4 bytes)

int main() {
    int client_fd;
    struct sockaddr_in server_addr;

    // Create a UDP socket
    client_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (client_fd < 0) {
        std::cerr << "Cannot open socket" << std::endl;
        return 1;
    }

    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(PORT);

    // Open the CSV file for input
    std::ifstream inputFile("X_test.csv");
    std::string line;

    // Open the CSV file for output
    std::ofstream outputFile("Results.csv");
    outputFile << "Memory Usage,Inference Time,Model Output\n"; // Header row

    // Read and send each line of the file, then wait for response
    while (std::getline(inputFile, line)) {
        // Send the line as a message
        if (sendto(client_fd, line.c_str(), line.length(), 0, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Failed to send message" << std::endl;
            close(client_fd);
            return 1;
        }

        // Now wait for the response
        unsigned char buffer[BUFFER_SIZE];
        struct sockaddr_in from;
        socklen_t fromLength = sizeof(from);

        int recvLength = recvfrom(client_fd, (char*)buffer, BUFFER_SIZE, 0, (struct sockaddr*)&from, &fromLength);
        if (recvLength > 0) {
            // Assuming server sends back 3 floats: memory usage, inference time, and model output
            float receivedResults[3];
            memcpy(receivedResults, buffer, sizeof(receivedResults));

            // Write to output CSV file
            outputFile << receivedResults[0] << "," << receivedResults[1] << "," << receivedResults[2] << "\n";
            outputFile.flush();

        } else {
            std::cerr << "Failed to receive response for " << line << std::endl;
        }

        usleep(10000); // 10 ms delay before sending the next message
    }

    std::cout << "All data processed" << std::endl;

    // Close the sockets and files
    close(client_fd);
    inputFile.close();
    outputFile.close();
    return 0;
}
