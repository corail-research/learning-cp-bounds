PYTHON_PATH="/Users/dariusdabert/anaconda3/bin/python3"
TORCHLIB_PATH="/Users/dariusdabert/Downloads/libtorch"

mknapsack:
	rm -rf src/problem/mknapsack/solving/build
	cmake -Hsrc/problem/mknapsack/solving -Bsrc/problem/mknapsack/solving/build -DCMAKE_PREFIX_PATH=$(TORCHLIB_PATH) -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/mknapsack/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/mknapsack/solving/build/solver_mknapsack ./

ssp:
	rm -rf src/problem/ssp/solving/build
	cmake -Hsrc/problem/ssp/solving -Bsrc/problem/ssp/solving/build -DCMAKE_PREFIX_PATH=$(TORCHLIB_PATH) -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/ssp/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/ssp/solving/build/solver_ssp ./
