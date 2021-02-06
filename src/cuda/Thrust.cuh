#include "Define.h"
#include "GPUmemory.h"

#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#ifndef _Thrust_H_
#define _Thrust_H_

template<typename TScan,typename T>
uint ThrustExclusiveScan(TScan* output, 
	T* input, uint numElements)
{
	thrust::exclusive_scan(
		thrust::device_ptr<T>(input),
		thrust::device_ptr<T>(input + numElements),
		thrust::device_ptr<TScan>(output));

	cudaDeviceSynchronize();

	T lastElement = 0;
	TScan lastElementScan = 0;

	checkCudaErrors(cudaMemcpy((void *)&lastElement, (void *)(input + numElements - 1),
		sizeof(T), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy((void *)&lastElementScan, (void *)(output + numElements - 1),
		sizeof(TScan), cudaMemcpyDeviceToHost));
	uint sum = lastElement + lastElementScan;

	return sum;
}

template<typename T>
void ThrustSort(T* particles, uint* key, uint numElements)
{
	thrust::device_ptr<T> posPtr(particles);
    thrust::sort_by_key(
		thrust::device_ptr<uint>(key),
		thrust::device_ptr<uint>(key + numElements),
		thrust::device_ptr<T> (posPtr));

	cudaDeviceSynchronize();
}

#endif