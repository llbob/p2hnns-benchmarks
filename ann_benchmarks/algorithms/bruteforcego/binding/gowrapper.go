package main

import (
	"fmt"
	"C"
	"unsafe"
	"MQH_thesis/algorithms/exhaustive"
	"MQH_thesis/types"
)

var rawData []float32
var pointIDs []int
var dimension int
var n int

//export index
func index(data *C.float, n C.int, d C.int) {
	numPoints = int(n)
	dimension = int(d)

	rawData = unsafe.Slice((*float32)(data), numPoints * dimension)
	pointIDs = make([]int, n) 
	for i := 0; i < n; i++ {
		pointIDs[i] = i
	}
	exhaustive.preprocess(rawData, numPoints, dimension)
}

//export query
func query(normal *C.float, bias C.float, k C.int, MinKList *C.int) *C.int {
	goNormal := unsafe.Slice((*float32)(normal), dimension)
	goBias := float32(bias)	
	goMinKList := unsafe.Slice((*int)(MinKList), int(k))

	exhaustive.search(rawData, goNormal, goBias, numPoints, dimension, int(k), goMinKList)

	// return original pointer that still points to the original memory that is now modified
	return MinKList
}

func main() {}