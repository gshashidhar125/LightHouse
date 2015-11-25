#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
using namespace std;
int main(int argc, char *argv[]) {

    long int numNodes, numEdges;
   
    if (argc != 3 || argv[1] == NULL || argv[2] == NULL) {
        printf("Wrong Number of Arguments\n");
        exit(1);
    }
    
    ifstream inputFile1;
    ifstream inputFile2;
    
    inputFile1.open(argv[1]);
    if (!inputFile1.is_open()){
        cout << "invalid file" << argv[1] << "\n";
        exit(1);
    }
    char temp[256];
    inputFile1.getline(temp, 256);
    inputFile1.getline(temp, 256);
    inputFile1.getline(temp, 256);
    inputFile1.getline(temp, 256);
    inputFile1.getline(temp, 256);
    
    inputFile2.open(argv[2]);
    if (!inputFile2.is_open()){
        cout << "invalid file" << argv[2] << "\n";
        exit(1);
    }
    
    long int numLine = 0;
    bool end = false, verify = true;
    long int node;
    while(!end) {
        long int dist1, dist2;
        inputFile1 >> dist1;
        inputFile2 >> node >> dist2;
        if (inputFile1.eof() || inputFile2.eof()) {
            end = true;
            break;
        }
        if (dist2 == 2147483647) {
            if (dist1 != 99999) {
                verify = false;
                end = true;
                break;
            }
        } else if (dist1 != dist2) {
            verify = false;
            end = true;
            break;
        }
        numLine++;
    }
    cout << "Line = " << numLine << "\n";
    cout << "Verify = " << verify << "\n";
    inputFile1.close();
    inputFile2.close();

}
