package main

import (
	"C"
	"MQH_thesis/pkg/algorithms"
	"unsafe"
)

var rawData []float32
var pointIDs []int
var dimension int
var numPoints int

//export Index
func Index(data *C.float, n C.int, d C.int) {
	numPoints = int(n)
	dimension = int(d)

	rawData = unsafe.Slice((*float32)(unsafe.Pointer(data)), numPoints*dimension)
	pointIDs = make([]int, int(n))
	for i := 0; i < int(n); i++ {
		pointIDs[i] = i
	}
	algorithms.Preprocess(rawData, numPoints, dimension)
}

//export Query
func Query(normal *C.float, bias C.float, k C.int, MinKList *C.int) *C.int {
	goNormal := unsafe.Slice((*float32)(unsafe.Pointer(normal)), dimension)
	goBias := float32(bias)
	goMinKList := unsafe.Slice((*int)(unsafe.Pointer(MinKList)), int(k))

	algorithms.Search(rawData, goNormal, goBias, numPoints, dimension, int(k), goMinKList)

	// return original pointer that still points to the original memory that is now modified
	return MinKList
}

func main() {}
