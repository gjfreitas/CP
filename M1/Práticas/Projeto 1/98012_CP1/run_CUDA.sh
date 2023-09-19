#!/bin/bash

echo -e "\nbuilding.pgm"
for i in {1..10}
do
./fastDetectorCuda -i building.pgm
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nchessBig.pgm"
for i in {1..10}
do
./fastDetectorCuda -i chessBig.pgm
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nchessRotate1.pgm"
for i in {1..10}
do
./fastDetectorCuda -i chessRotate1.pgm
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nhouse.pgm"
for i in {1..10}
do
./fastDetectorCuda -i house.pgm
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nsquares.pgm"
for i in {1..10}
do
./fastDetectorCuda -i squares.pgm
./testDiffs referenceCuda.pgm resultCuda.pgm
done