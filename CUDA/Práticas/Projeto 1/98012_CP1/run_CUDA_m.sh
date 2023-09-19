#!/bin/bash

echo -e "\nbuilding.pgm"
for i in {1..10}
do
./fastDetectorCuda -i building.pgm -m
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nchessBig.pgm"
for i in {1..10}
do
./fastDetectorCuda -i chessBig.pgm -m
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nchessRotate1.pgm"
for i in {1..10}
do
./fastDetectorCuda -i chessRotate1.pgm -m
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nhouse.pgm"
for i in {1..10}
do
./fastDetectorCuda -i house.pgm -m
./testDiffs referenceCuda.pgm resultCuda.pgm
done

echo -e "\nsquares.pgm"
for i in {1..10}
do
./fastDetectorCuda -i squares.pgm -m
./testDiffs referenceCuda.pgm resultCuda.pgm
done