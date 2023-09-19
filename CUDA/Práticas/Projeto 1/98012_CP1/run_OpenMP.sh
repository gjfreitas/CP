#!/bin/bash

echo -e "\nbuilding.pgm"
for i in {1..10}
do
./fastDetectorOpenMP -i building.pgm
./testDiffs referenceOpenMP.pgm resultOpenMP.pgm
done

echo -e "\nchessBig.pgm"
for i in {1..10}
do
./fastDetectorOpenMP -i chessBig.pgm
./testDiffs referenceOpenMP.pgm resultOpenMP.pgm
done

echo -e "\nchessRotate1.pgm"
for i in {1..10}
do
./fastDetectorOpenMP -i chessRotate1.pgm
./testDiffs referenceOpenMP.pgm resultOpenMP.pgm
done

echo -e "\nhouse.pgm"
for i in {1..10}
do
./fastDetectorOpenMP -i house.pgm
./testDiffs referenceOpenMP.pgm resultOpenMP.pgm
done

echo -e "\nsquares.pgm"
for i in {1..10}
do
./fastDetectorOpenMP -i squares.pgm
./testDiffs referenceOpenMP.pgm resultOpenMP.pgm
done